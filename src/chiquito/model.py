from __future__ import annotations

from typing import Optional, Union
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    GenerationMixin,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device

from .utils import clean_memory, load_safetensors
from .splitter import find_or_create_split, layer_file_path


class ChiquitoModel(GenerationMixin):

    LAYER_NAMES = {
        "embed": "model.embed_tokens",
        "layer_prefix": "model.layers",
        "norm": "model.norm",
        "lm_head": "lm_head",
    }

    def __init__(
        self,
        model_id_or_path: str,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
        max_seq_len: int = 512,
        preload_to_ram: bool = True,
        hf_token: str | None = None,
        prefetch: bool = True,
    ):
        self._device = torch.device(device)
        self._dtype = dtype
        self.max_seq_len = max_seq_len
        self._preload_to_ram = preload_to_ram
        self._hf_token = hf_token
        self._prefetch = prefetch
        self._ram_cache: dict[str, dict[str, torch.Tensor]] | None = None
        self.hf_quantizer = None
        self.main_input_name = "input_ids"

        # Build layer name list before splitting
        all_layer_names = self._build_layer_name_list_from_config(model_id_or_path, hf_token)

        # Split model into per-layer files
        self._model_path, self._split_dir = find_or_create_split(
            model_id_or_path, all_layer_names, hf_token=hf_token
        )

        # Load config, tokenizer, generation config
        token_kwargs = {"token": hf_token} if hf_token else {}
        self.config = AutoConfig.from_pretrained(
            self._model_path, trust_remote_code=True, **token_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._model_path, trust_remote_code=True, **token_kwargs
        )
        try:
            self.generation_config = GenerationConfig.from_pretrained(self._model_path)
        except Exception:
            self.generation_config = GenerationConfig()

        # Create meta model and build layer references
        self._init_model()

        # Preload weights to RAM if requested
        if preload_to_ram:
            self._preload_all_layers()

    def _build_layer_name_list_from_config(
        self, model_id_or_path: str, hf_token: str | None
    ) -> list[str]:
        token_kwargs = {"token": hf_token} if hf_token else {}
        config = AutoConfig.from_pretrained(
            model_id_or_path, trust_remote_code=True, **token_kwargs
        )
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

        # Count layers
        module = model
        for attr in self.LAYER_NAMES["layer_prefix"].split("."):
            module = getattr(module, attr)
        n_layers = len(module)
        del model

        names = self.LAYER_NAMES
        return (
            [names["embed"]]
            + [f'{names["layer_prefix"]}.{i}' for i in range(n_layers)]
            + [names["norm"], names["lm_head"]]
        )

    def _init_model(self):
        self.model = None

        # Try sdpa attention first
        try:
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(
                    self.config, attn_implementation="sdpa", trust_remote_code=True
                )
        except (TypeError, ValueError):
            self.model = None

        # Fallback
        if self.model is None:
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_config(
                    self.config, trust_remote_code=True
                )

        # Handle HF quantization config if present
        quantization_config = getattr(self.config, "quantization_config", None)
        if quantization_config is not None:
            from transformers.quantizers import AutoHfQuantizer

            self.hf_quantizer = AutoHfQuantizer.from_config(
                quantization_config, pre_quantized=True
            )
            device_map = self.hf_quantizer.update_device_map(None)
            self.hf_quantizer.preprocess_model(model=self.model, device_map=device_map)

        self.model.eval()
        self.model.tie_weights()

        # Build layer lists
        self._build_layers()

        # Move buffers to device
        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(
                self.model, buffer_name, self._device, value=buffer, dtype=self._dtype
            )

    def _build_layers(self):
        names = self.LAYER_NAMES
        self.layers: list[nn.Module] = []
        self.layer_names: list[str] = []

        # Embedding
        module = self.model
        for attr in names["embed"].split("."):
            module = getattr(module, attr)
        self.layers.append(module)
        self.layer_names.append(names["embed"])

        # Transformer layers
        module = self.model
        for attr in names["layer_prefix"].split("."):
            module = getattr(module, attr)
        for i, layer in enumerate(module):
            self.layers.append(layer)
            self.layer_names.append(f'{names["layer_prefix"]}.{i}')

        # Norm
        module = self.model
        for attr in names["norm"].split("."):
            module = getattr(module, attr)
        self.layers.append(module)
        self.layer_names.append(names["norm"])

        # LM head
        module = self.model
        for attr in names["lm_head"].split("."):
            module = getattr(module, attr)
        self.layers.append(module)
        self.layer_names.append(names["lm_head"])

    def _preload_all_layers(self):
        self._ram_cache = {}
        for name in tqdm(self.layer_names, desc="Preloading layers to RAM"):
            self._ram_cache[name] = load_safetensors(
                layer_file_path(self._split_dir, name)
            )

    def _load_layer_to_cpu(self, layer_name: str) -> dict[str, torch.Tensor]:
        if self._ram_cache is not None:
            return self._ram_cache[layer_name]
        return load_safetensors(layer_file_path(self._split_dir, layer_name))

    def _move_layer_to_device(self, state_dict: dict[str, torch.Tensor]) -> list[str]:
        moved = []
        for param_name in state_dict:
            if self.hf_quantizer is None:
                moved.append(param_name)
            else:
                if ".weight" in param_name:
                    layer_name = param_name[: param_name.index(".weight") + len(".weight")]
                    if layer_name not in moved:
                        moved.append(layer_name)

        for param_name in moved:
            if self.hf_quantizer is None or not self.hf_quantizer.check_quantized_param(
                self.model, param_value=None, param_name=param_name, state_dict={}
            ):
                set_module_tensor_to_device(
                    self.model,
                    param_name,
                    self._device,
                    value=state_dict[param_name],
                    dtype=self._dtype,
                )
            else:
                self.hf_quantizer.create_quantized_param(
                    self.model, state_dict[param_name], param_name, self._device, state_dict
                )
        return moved

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    # --- Override points for subclasses ---

    def _compute_position_embeddings(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        rotary_emb = getattr(getattr(self.model, "model", None), "rotary_emb", None)
        if rotary_emb is not None:
            return rotary_emb(hidden_states, position_ids=position_ids)
        return None

    def _run_transformer_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        kwargs: dict = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": False,
        }
        if position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings
        out = layer(hidden_states, **kwargs)
        return out[0] if isinstance(out, tuple) else out

    def _run_norm(self, layer: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        return layer(hidden_states)

    def _run_lm_head(self, layer: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        return layer(hidden_states).float()

    # --- GenerationMixin interface ---

    def can_generate(self) -> bool:
        return True

    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=None, **kwargs
    ):
        position_ids = None
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # --- Forward pass ---

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, CausalLMOutputWithPast]:
        # Reinit model to ensure clean buffer state
        del self.model
        clean_memory()
        self._init_model()

        input_ids = input_ids.to(self._device)
        seq_len = input_ids.shape[1]

        # Causal attention mask: (1, 1, S, S)
        causal_mask = torch.ones(seq_len, seq_len, device=self._device)
        causal_mask = causal_mask.triu(diagonal=1)[None, None, ...] == 0
        causal_mask = causal_mask.to(self._device)

        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=self._device)[None, :]

        hidden_states = None
        position_embeddings = None
        names = self.LAYER_NAMES

        with torch.inference_mode():
            executor = ThreadPoolExecutor(max_workers=1) if self._prefetch else None
            future = None

            if executor:
                future = executor.submit(self._load_layer_to_cpu, self.layer_names[0])

            for i, (layer_name, layer) in tqdm(
                enumerate(zip(self.layer_names, self.layers)),
                total=len(self.layers),
                desc=f"Running layers ({self._device})",
            ):
                # Load weights
                if executor and future:
                    state_dict = future.result()
                    if i + 1 < len(self.layer_names):
                        future = executor.submit(
                            self._load_layer_to_cpu, self.layer_names[i + 1]
                        )
                else:
                    state_dict = self._load_layer_to_cpu(layer_name)

                moved = self._move_layer_to_device(state_dict)

                # Run layer
                if layer_name == names["embed"]:
                    hidden_states = layer(input_ids)
                    position_embeddings = self._compute_position_embeddings(
                        hidden_states, position_ids
                    )
                elif layer_name == names["norm"]:
                    hidden_states = self._run_norm(layer, hidden_states)
                elif layer_name == names["lm_head"]:
                    hidden_states = self._run_lm_head(layer, hidden_states)
                else:
                    hidden_states = self._run_transformer_layer(
                        layer,
                        hidden_states,
                        attention_mask=causal_mask[:, :, :seq_len, :seq_len],
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )

                # Free GPU memory
                if self.hf_quantizer is not None:
                    for param_name in moved:
                        set_module_tensor_to_device(self.model, param_name, "meta")
                else:
                    layer.to("meta")
                layer.to("meta")
                clean_memory()

            if executor:
                executor.shutdown(wait=False)

        logits = hidden_states
        return CausalLMOutputWithPast(logits=logits)

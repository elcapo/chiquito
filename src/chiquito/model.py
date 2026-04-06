from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    GenerationMixin,
)
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from .splitter import find_or_create_split, layer_file_path
from .utils import clean_gpu_memory, clean_memory, load_safetensors


class _SlidingWindowCache:
    """Bounded RAM cache that keeps at most `window_size` layers loaded,
    with a background thread that stays ahead of consumption."""

    def __init__(
        self,
        layer_names: list[str],
        split_dir: Path,
        window_size: int,
    ):
        self._layer_names = layer_names
        self._split_dir = split_dir
        self._window_size = window_size
        self._cache: dict[str, dict[str, torch.Tensor]] = {}
        self._lock = threading.Lock()
        self._ready = threading.Condition(self._lock)
        self._loader_thread: threading.Thread | None = None
        self._next_to_load = 0
        self._stop = False

    def start(self):
        initial_count = min(self._window_size, len(self._layer_names))
        for i in tqdm(range(initial_count), desc="Preloading window to RAM"):
            name = self._layer_names[i]
            self._cache[name] = load_safetensors(layer_file_path(self._split_dir, name))
        self._next_to_load = initial_count
        self._loader_thread = threading.Thread(
            target=self._background_loader, daemon=True
        )
        self._loader_thread.start()

    def get(self, layer_name: str) -> dict[str, torch.Tensor]:
        with self._ready:
            while layer_name not in self._cache:
                self._ready.wait()
            return self._cache[layer_name]

    def release(self, layer_name: str):
        with self._ready:
            self._cache.pop(layer_name, None)
            self._ready.notify_all()

    def stop(self):
        with self._ready:
            self._stop = True
            self._ready.notify_all()
        if self._loader_thread is not None:
            self._loader_thread.join(timeout=5)

    def _background_loader(self):
        while True:
            with self._ready:
                if self._stop or self._next_to_load >= len(self._layer_names):
                    return
                while len(self._cache) >= self._window_size:
                    if self._stop:
                        return
                    self._ready.wait()
                name = self._layer_names[self._next_to_load]
                self._next_to_load += 1

            data = load_safetensors(layer_file_path(self._split_dir, name))

            with self._ready:
                self._cache[name] = data
                self._ready.notify_all()


class ChiquitoModel(GenerationMixin):
    LAYER_NAMES: ClassVar[dict] = {
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
        preload_to_ram: bool | int = True,
        hf_token: str | None = None,
        prefetch: bool = True,
        quantization: str | None = None,
    ):
        self._device = torch.device(device)
        self._dtype = dtype
        self.max_seq_len = max_seq_len
        self._hf_token = hf_token
        self._prefetch = prefetch
        self._quantization = quantization
        self._ram_cache: dict[str, dict[str, torch.Tensor]] | None = None
        self._window_cache: _SlidingWindowCache | None = None
        self._window_size: int | None = None
        self.hf_quantizer: Any = None
        self.main_input_name = "input_ids"

        # Interpret preload_to_ram
        if preload_to_ram is True:
            self._window_size = None  # load all
        elif preload_to_ram is False:
            self._window_size = 0  # disk only
        else:
            self._window_size = int(preload_to_ram)

        # Build layer name list before splitting
        all_layer_names = self._build_layer_name_list_from_config(
            model_id_or_path, hf_token
        )

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

        # Preload weights to RAM
        if self._window_size is None:
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
            + [f"{names['layer_prefix']}.{i}" for i in range(n_layers)]
            + [names["norm"], names["lm_head"]]
        )

    def _init_model(self):
        self.model: Any = None

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

        # Handle quantization
        quantization_config = getattr(self.config, "quantization_config", None)
        if self._quantization is not None:
            from transformers import BitsAndBytesConfig
            from transformers.quantizers import AutoHfQuantizer

            if self._quantization == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self._dtype,
                    bnb_4bit_quant_type="nf4",
                )
            elif self._quantization == "8bit":
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise ValueError(
                    f"Unknown quantization: {self._quantization!r}. Use '4bit' or '8bit'."
                )

            self.hf_quantizer = AutoHfQuantizer.from_config(bnb_config)
            device_map = self.hf_quantizer.update_device_map(None)
            self.hf_quantizer.preprocess_model(model=self.model, device_map=device_map)
        elif quantization_config is not None:
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

    def _reset_model_to_meta(self):
        """Move all parameters back to meta device without recreating the model."""
        for name, _ in self.model.named_parameters():
            set_module_tensor_to_device(self.model, name, "meta")
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
            self.layer_names.append(f"{names['layer_prefix']}.{i}")

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
        pin = torch.cuda.is_available()
        for name in tqdm(self.layer_names, desc="Preloading layers to RAM"):
            state_dict = load_safetensors(layer_file_path(self._split_dir, name))
            if pin:
                state_dict = {k: v.pin_memory() for k, v in state_dict.items()}
            self._ram_cache[name] = state_dict

    def _start_window_cache(self):
        if self._window_cache is not None:
            self._window_cache.stop()
        assert self._window_size is not None
        self._window_cache = _SlidingWindowCache(
            self.layer_names, self._split_dir, self._window_size
        )
        self._window_cache.start()

    def _load_layer_to_cpu(self, layer_name: str) -> dict[str, torch.Tensor]:
        if self._ram_cache is not None:
            return self._ram_cache[layer_name]
        if self._window_cache is not None:
            return self._window_cache.get(layer_name)
        return load_safetensors(layer_file_path(self._split_dir, layer_name))

    def _move_layer_to_device(self, state_dict: dict[str, torch.Tensor]) -> list[str]:
        moved = list(state_dict.keys())
        for param_name in moved:
            set_module_tensor_to_device(
                self.model,
                param_name,
                self._device,
                value=state_dict[param_name],
                dtype=self._dtype,
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
            result: tuple[torch.Tensor, torch.Tensor] = rotary_emb(
                hidden_states, position_ids=position_ids
            )
            return result
        return None

    def _run_transformer_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        past_key_values: DynamicCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        kwargs: dict = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": use_cache,
        }
        if position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings
        if past_key_values is not None:
            kwargs["past_key_values"] = past_key_values
        out = layer(hidden_states, **kwargs)
        return out[0] if isinstance(out, tuple) else out

    def _run_norm(self, layer: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        return layer(hidden_states)

    def _run_lm_head(
        self, layer: nn.Module, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return layer(hidden_states).float()

    # --- GenerationMixin interface ---

    def can_generate(self) -> bool:
        return True

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        past_key_values = kwargs.get("past_key_values")

        if past_key_values is not None:
            past_len = past_key_values.get_seq_length()
            if input_ids.shape[1] > past_len:
                input_ids = input_ids[:, past_len:]
            else:
                input_ids = input_ids[:, -1:]

        position_ids = None
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
        }

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # --- Forward pass ---

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> tuple | CausalLMOutputWithPast:
        # Reset model to clean state
        if self.hf_quantizer is not None:
            # Quantized models need full reinit (bnb modules can't round-trip to meta)
            del self.model
            clean_memory()
            self._init_model()
        else:
            self._reset_model_to_meta()
            clean_gpu_memory()

        # Restart sliding window cache for this forward pass
        if self._window_size is not None and self._window_size > 0:
            self._start_window_cache()

        input_ids = input_ids.to(self._device)
        seq_len = input_ids.shape[1]

        # KV cache setup
        if past_key_values is None:
            past_key_values = DynamicCache()
        past_len = past_key_values.get_seq_length()
        is_prefill = past_len == 0
        total_len = past_len + seq_len

        # Causal attention mask spanning cached + new tokens
        if is_prefill:
            causal_mask = torch.ones(seq_len, seq_len, device=self._device)
            causal_mask = causal_mask.triu(diagonal=1)[None, None, ...] == 0
        else:
            # Decode: new token(s) attend to all previous + themselves
            causal_mask = torch.ones(
                1, 1, seq_len, total_len, dtype=torch.bool, device=self._device
            )

        if position_ids is None:
            position_ids = torch.arange(
                past_len, total_len, dtype=torch.long, device=self._device
            )[None, :]

        hidden_states = None
        position_embeddings = None
        names = self.LAYER_NAMES

        # Use ThreadPoolExecutor prefetch only when NOT using sliding window
        use_executor = self._prefetch and self._window_cache is None

        with torch.inference_mode():
            executor = ThreadPoolExecutor(max_workers=1) if use_executor else None
            future = None

            if executor:
                future = executor.submit(self._load_layer_to_cpu, self.layer_names[0])

            for i, (layer_name, layer) in tqdm(
                enumerate(zip(self.layer_names, self.layers, strict=True)),
                total=len(self.layers),
                desc=f"Running layers ({self._device})",
                disable=False,
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
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                # Free GPU memory (weights only, not KV cache)
                for param_name in moved:
                    set_module_tensor_to_device(self.model, param_name, "meta")
                clean_gpu_memory()

                # Release layer from sliding window cache
                if self._window_cache is not None:
                    self._window_cache.release(layer_name)

            if executor:
                executor.shutdown(wait=False)
            if self._window_cache is not None:
                self._window_cache.stop()

        logits = hidden_states
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

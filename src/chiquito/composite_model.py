from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn as nn
from accelerate.utils.modeling import set_module_tensor_to_device

from .lazy_experts import LazyDequantExperts
from .model import ChiquitoModel
from .splitter import parse_quantized_state_dict
from .utils import clean_gpu_memory


class ChiquitoCompositeModel(ChiquitoModel):
    """Layer-by-layer inference for composite / multimodal HuggingFace models.

    Many vision-language models (Qwen3.5-MoE, LLaVA, InternVL, …) store the
    text-model weights under ``model.language_model.*`` on disk, while
    ``AutoModelForCausalLM.from_config(text_config)`` produces a model whose
    attributes live directly under ``model.*``.  This subclass bridges that
    gap so the splitter sees the on-disk prefixes and the runtime sees the
    model-object paths.
    """

    # On-disk weight key prefixes (used by the splitter and for file naming).
    LAYER_NAMES: ClassVar[dict] = {
        "embed": "model.language_model.embed_tokens",
        "layer_prefix": "model.language_model.layers",
        "norm": "model.language_model.norm",
        "lm_head": "lm_head",
    }

    # Model-object attribute paths (for getattr traversal on the meta model).
    _MODEL_LAYER_NAMES: ClassVar[dict] = {
        "embed": "model.embed_tokens",
        "layer_prefix": "model.layers",
        "norm": "model.norm",
        "lm_head": "lm_head",
    }

    _DISK_PREFIX = "model.language_model."
    _MODEL_PREFIX = "model."

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _build_layers(self) -> None:
        """Traverse the model object with ``_MODEL_LAYER_NAMES`` but store
        ``LAYER_NAMES`` entries in ``self.layer_names`` for file loading."""
        # Force eager expert dispatch so the MoE forward accesses expert
        # weights via scalar indexing (wrapper[i]), not full-tensor ops.
        config = getattr(self, "config", None)
        if config is not None:
            if hasattr(config, "_experts_implementation"):
                config._experts_implementation = "eager"
            if hasattr(config, "_experts_implementation_internal"):
                config._experts_implementation_internal = "eager"

        model_names = self._MODEL_LAYER_NAMES
        disk_names = self.LAYER_NAMES
        self.layers: list[nn.Module] = []
        self.layer_names: list[str] = []

        # Embedding
        module = self.model
        for attr in model_names["embed"].split("."):
            module = getattr(module, attr)
        self.layers.append(module)
        self.layer_names.append(disk_names["embed"])

        # Transformer layers
        module = self.model
        for attr in model_names["layer_prefix"].split("."):
            module = getattr(module, attr)
        for i, layer in enumerate(module):
            self.layers.append(layer)
            self.layer_names.append(f"{disk_names['layer_prefix']}.{i}")

        # Norm
        module = self.model
        for attr in model_names["norm"].split("."):
            module = getattr(module, attr)
        self.layers.append(module)
        self.layer_names.append(disk_names["norm"])

        # LM head
        module = self.model
        for attr in model_names["lm_head"].split("."):
            module = getattr(module, attr)
        self.layers.append(module)
        self.layer_names.append(disk_names["lm_head"])

    def _remap_state_dict_keys(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Rename ``model.language_model.*`` keys to ``model.*``."""
        remapped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith(self._DISK_PREFIX):
                new_key = self._MODEL_PREFIX + key[len(self._DISK_PREFIX) :]
                remapped[new_key] = value
            else:
                remapped[key] = value
        return remapped

    def _get_quantizable_param_names(
        self, model_id_or_path: str, hf_token: str | None
    ) -> set[str] | None:
        """Return quantizable param names remapped to on-disk format."""
        model_names = super()._get_quantizable_param_names(model_id_or_path, hf_token)
        if model_names is None:
            return None
        # model-object → disk: model.X → model.language_model.X
        remapped: set[str] = set()
        for name in model_names:
            if name.startswith(self._MODEL_PREFIX):
                remapped.add(self._DISK_PREFIX + name[len(self._MODEL_PREFIX) :])
            else:
                remapped.add(name)
        return remapped

    # ------------------------------------------------------------------
    # Lazy expert dequantization for MoE fused 3-D weight tensors
    # ------------------------------------------------------------------

    def _move_quantized_layer_to_device(
        self, raw_state_dict: dict[str, torch.Tensor]
    ) -> list[str]:
        """Override to use ``LazyDequantExperts`` for fused 3-D expert tensors.

        Standard 2-D ``nn.Linear`` weights are handled normally via ``super()``.
        Fused expert weights (detected by ``original_shape`` in their quant
        state) are wrapped in a ``LazyDequantExperts`` and set as plain
        attributes on the module — **not** as ``nn.Parameter``.
        """
        base_params, qs_map = parse_quantized_state_dict(raw_state_dict)

        # Separate fused expert tensors from everything else
        lazy_names: list[str] = []
        regular_sd: dict[str, torch.Tensor] = {}

        for name, tensor in base_params.items():
            if (
                name in qs_map
                and "original_shape" in qs_map[name]
                and ".experts." in name
            ):
                # Fused MoE expert tensor — build lazy wrapper
                wrapper = LazyDequantExperts.from_quantized(
                    packed=tensor,
                    qs_entries=qs_map.pop(name),
                    dtype=self._dtype,
                    device=self._device,
                )
                attr_parts = name.split(".")
                module: Any = self.model
                for s in attr_parts[:-1]:
                    module = getattr(module, s)
                attr_name = attr_parts[-1]
                if attr_name in module._parameters:
                    del module._parameters[attr_name]
                setattr(module, attr_name, wrapper)
                lazy_names.append(name)
            else:
                # Re-pack into a state_dict for the regular path
                regular_sd[name] = tensor
                if name in qs_map:
                    for qs_key, qs_val in qs_map[name].items():
                        regular_sd[f"{name}.{qs_key}"] = qs_val

        # Delegate 2-D params to base class
        moved = super()._move_quantized_layer_to_device(regular_sd)
        return moved + lazy_names

    def _cleanup_moved_params(self, moved: list[str]) -> None:
        """Handle cleanup of both regular params and lazy expert wrappers."""
        for param_name in moved:
            parts = param_name.split(".")
            module: Any = self.model
            for s in parts[:-1]:
                module = getattr(module, s)
            attr_name = parts[-1]
            current = getattr(module, attr_name, None)

            if isinstance(current, LazyDequantExperts):
                # Replace lazy wrapper with an empty meta parameter
                delattr(module, attr_name)
                module._parameters[attr_name] = torch.nn.Parameter(
                    torch.empty(current.shape, device="meta"),
                    requires_grad=False,
                )
            else:
                set_module_tensor_to_device(self.model, param_name, "meta")

        clean_gpu_memory()

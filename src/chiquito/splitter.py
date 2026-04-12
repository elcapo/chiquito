import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from tqdm import tqdm

from .utils import clean_memory, load_safetensors, resolve_model_path, save_safetensors

SPLIT_DIR_NAME = "chiquito_split"

# Suffixes produced by QuantState.as_dict(packed=True).
# Used to tell quant-state entries apart from base parameter tensors.
_QUANT_STATE_SUFFIXES = (
    ".absmax",
    ".quant_map",
    ".quant_state.",
    ".nested_absmax",
    ".nested_quant_map",
)


def split_dir_name(quantization: str | None = None) -> str:
    """Return the split directory name, with a quantization suffix when applicable."""
    if quantization:
        return f"{SPLIT_DIR_NAME}_{quantization}"
    return SPLIT_DIR_NAME


def layer_file_path(split_dir: Path, layer_name: str) -> Path:
    return split_dir / (layer_name + ".safetensors")


def done_marker_path(split_dir: Path, layer_name: str) -> Path:
    return split_dir / (layer_name + ".safetensors.done")


def is_layer_split(split_dir: Path, layer_name: str) -> bool:
    return (
        layer_file_path(split_dir, layer_name).exists()
        and done_marker_path(split_dir, layer_name).exists()
    )


def _quantize_state_dict(
    state_dict: dict[str, torch.Tensor],
    quantization: str,
    device: str = "cuda:0",
) -> dict[str, torch.Tensor]:
    """Quantize 2-D+ weight tensors and return packed data + quant-state entries.

    Supports ``"4bit"`` (NF4) and ``"8bit"`` (block-wise absmax).
    """
    import bitsandbytes.functional as F

    result: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        if tensor.ndim < 2:
            result[name] = tensor
            continue

        gpu_tensor = tensor.to(torch.float16).to(device).contiguous()
        if quantization == "4bit":
            packed, quant_state = F.quantize_4bit(
                gpu_tensor, quant_type="nf4", blocksize=64
            )
        elif quantization == "8bit":
            packed, quant_state = F.quantize_blockwise(gpu_tensor)
        else:
            raise ValueError(
                f"Unknown quantization: {quantization!r}. Use '4bit' or '8bit'."
            )

        result[name] = packed.cpu()
        for qs_key, qs_val in quant_state.as_dict(packed=True).items():
            result[f"{name}.{qs_key}"] = (
                qs_val.cpu() if isinstance(qs_val, torch.Tensor) else qs_val
            )
        del gpu_tensor, packed, quant_state

    return result


def parse_quantized_state_dict(
    raw: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    """Separate base parameter tensors from quant-state entries.

    Returns:
        base_params: ``{param_name: tensor}`` — packed weights, biases, norms, etc.
        quant_states: ``{param_name: {qs_suffix: tensor}}`` — only for quantized weights.
    """
    base: dict[str, torch.Tensor] = {}
    qs_map: dict[str, dict[str, torch.Tensor]] = {}

    for key, val in raw.items():
        parent: str | None = None
        suffix: str | None = None
        for qs_sfx in _QUANT_STATE_SUFFIXES:
            idx = key.find(qs_sfx)
            if idx != -1:
                parent = key[:idx]
                suffix = key[idx + 1 :]  # strip leading dot
                break

        if parent is not None and suffix is not None:
            qs_map.setdefault(parent, {})[suffix] = val
        else:
            base[key] = val

    return base, qs_map


def split_and_save_layers(
    model_path: Path,
    layer_names: list[str],
    hf_token: str | None = None,
    repo_id: str | None = None,
    quantization: str | None = None,
) -> Path:
    dir_name = split_dir_name(quantization)
    split_dir = model_path / dir_name

    # Check if all layers already split (/ quantized)
    if split_dir.exists() and all(
        is_layer_split(split_dir, name) for name in layer_names
    ):
        print(f"All layers already split in {split_dir}")
        return split_dir

    # Quantized splits are built from the base (fp16) split
    if quantization is not None:
        base_dir = model_path / SPLIT_DIR_NAME
        if not (
            base_dir.exists() and all(is_layer_split(base_dir, n) for n in layer_names)
        ):
            split_and_save_layers(
                model_path, layer_names, hf_token=hf_token, repo_id=repo_id
            )

        split_dir.mkdir(parents=True, exist_ok=True)
        # Only quantize decoder layers (model.layers.*).  Non-decoder layers
        # (embed_tokens, norm, lm_head) are kept in fp16 — they are small, and
        # HF quantizers deliberately skip them (modules_to_not_convert).
        _DECODER_PREFIX = "model.layers."
        for layer_name in tqdm(layer_names, desc=f"Quantizing layers ({quantization})"):
            if is_layer_split(split_dir, layer_name):
                continue
            fp16_data = load_safetensors(layer_file_path(base_dir, layer_name))
            if layer_name.startswith(_DECODER_PREFIX):
                quantized = _quantize_state_dict(fp16_data, quantization)
            else:
                quantized = fp16_data
            save_safetensors(quantized, layer_file_path(split_dir, layer_name))
            done_marker_path(split_dir, layer_name).touch()
            del fp16_data, quantized
            clean_memory()

        return split_dir

    # --- Non-quantized (fp16) split from original model files ---
    split_dir.mkdir(parents=True, exist_ok=True)

    # Load weight map
    index_path = model_path / "model.safetensors.index.json"
    single_file = model_path / "model.safetensors"

    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    elif single_file.exists():
        # Single-file model: build a synthetic weight map
        weight_map = None
    else:
        raise FileNotFoundError(
            f"No model.safetensors.index.json or model.safetensors found in {model_path}"
        )

    state_dict: dict = {}

    if weight_map is None:
        # Single safetensors file: load everything, split, save
        state_dict = load_file(str(single_file), device="cpu")
        for layer_name in tqdm(layer_names, desc="Splitting layers"):
            if is_layer_split(split_dir, layer_name):
                continue
            prefix = layer_name + "."
            layer_state = {k: v for k, v in state_dict.items() if k.startswith(prefix)}
            if layer_state:
                save_safetensors(layer_state, layer_file_path(split_dir, layer_name))
                done_marker_path(split_dir, layer_name).touch()
        del state_dict
        clean_memory()
        return split_dir

    # Multi-shard model: load shards incrementally
    # Figure out which shards exist and their numbering
    loaded_shards: set[str] = set()
    state_dict = {}

    for layer_name in tqdm(layer_names, desc="Splitting layers"):
        if is_layer_split(split_dir, layer_name):
            continue

        prefix = layer_name + "."
        # Find which shard files contain weights for this layer
        needed_shards = {
            shard for param, shard in weight_map.items() if param.startswith(prefix)
        }

        # Load any shards we haven't loaded yet
        for shard in needed_shards:
            if shard not in loaded_shards:
                shard_path = model_path / shard
                if not shard_path.exists() and repo_id:
                    import huggingface_hub

                    huggingface_hub.snapshot_download(
                        repo_id,
                        allow_patterns=[shard],
                        token=hf_token,
                    )
                state_dict.update(load_file(str(shard_path), device="cpu"))
                loaded_shards.add(shard)

        # Extract this layer's weights
        layer_state = {k: v for k, v in state_dict.items() if k.startswith(prefix)}
        if layer_state:
            save_safetensors(layer_state, layer_file_path(split_dir, layer_name))
            done_marker_path(split_dir, layer_name).touch()

            # Free extracted weights from state_dict
            for k in layer_state:
                del state_dict[k]
            del layer_state
            clean_memory()

    return split_dir


def find_or_create_split(
    model_id_or_path: str,
    layer_names: list[str],
    hf_token: str | None = None,
    quantization: str | None = None,
) -> tuple[Path, Path]:
    model_path = resolve_model_path(model_id_or_path, hf_token)

    # Determine repo_id for lazy shard downloading
    repo_id = None if Path(model_id_or_path).is_dir() else model_id_or_path

    split_dir = split_and_save_layers(
        model_path,
        layer_names,
        hf_token=hf_token,
        repo_id=repo_id,
        quantization=quantization,
    )
    return model_path, split_dir

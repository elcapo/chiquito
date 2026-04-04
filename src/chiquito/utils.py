import gc
import ctypes
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
import huggingface_hub


def clean_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_safetensors(path: Path) -> dict[str, torch.Tensor]:
    return load_file(str(path), device="cpu")


def save_safetensors(state_dict: dict[str, torch.Tensor], path: Path):
    save_file(state_dict, str(path))


def resolve_model_path(model_id_or_path: str, hf_token: str | None = None) -> Path:
    path = Path(model_id_or_path)
    if path.is_dir():
        if (path / "model.safetensors.index.json").exists() or (path / "model.safetensors").exists():
            return path
        print(f"Local directory {path} exists but no model files found, treating as HF repo ID...")

    cache_path = huggingface_hub.snapshot_download(
        model_id_or_path,
        token=hf_token,
        ignore_patterns=["*.bin"],
    )
    return Path(cache_path)

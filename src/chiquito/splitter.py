import json
from pathlib import Path

from tqdm import tqdm
from safetensors.torch import load_file

from .utils import save_safetensors, clean_memory, resolve_model_path


SPLIT_DIR_NAME = "chiquito_split"


def layer_file_path(split_dir: Path, layer_name: str) -> Path:
    return split_dir / (layer_name + ".safetensors")


def done_marker_path(split_dir: Path, layer_name: str) -> Path:
    return split_dir / (layer_name + ".safetensors.done")


def is_layer_split(split_dir: Path, layer_name: str) -> bool:
    return layer_file_path(split_dir, layer_name).exists() and done_marker_path(split_dir, layer_name).exists()


def split_and_save_layers(
    model_path: Path,
    layer_names: list[str],
    hf_token: str | None = None,
    repo_id: str | None = None,
) -> Path:
    split_dir = model_path / SPLIT_DIR_NAME

    # Check if all layers already split
    if split_dir.exists() and all(is_layer_split(split_dir, name) for name in layer_names):
        print(f"All layers already split in {split_dir}")
        return split_dir

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
    state_dict: dict = {}

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
) -> tuple[Path, Path]:
    model_path = resolve_model_path(model_id_or_path, hf_token)

    # Determine repo_id for lazy shard downloading
    repo_id = None if Path(model_id_or_path).is_dir() else model_id_or_path

    split_dir = split_and_save_layers(model_path, layer_names, hf_token=hf_token, repo_id=repo_id)
    return model_path, split_dir

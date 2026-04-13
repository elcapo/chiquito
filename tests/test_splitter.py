import json
from pathlib import Path
from unittest.mock import patch

import torch

from chiquito.splitter import (
    SPLIT_DIR_NAME,
    done_marker_path,
    find_or_create_split,
    is_layer_split,
    layer_file_path,
    parse_quantized_state_dict,
    split_and_save_layers,
    split_dir_name,
)
from chiquito.utils import save_safetensors


class TestPathHelpers:
    def test_layer_file_path(self, tmp_path):
        result = layer_file_path(tmp_path, "model.layers.0")
        assert result == tmp_path / "model.layers.0.safetensors"

    def test_done_marker_path(self, tmp_path):
        result = done_marker_path(tmp_path, "model.layers.0")
        assert result == tmp_path / "model.layers.0.safetensors.done"


class TestIsLayerSplit:
    def test_returns_false_when_nothing_exists(self, tmp_path):
        assert not is_layer_split(tmp_path, "layer")

    def test_returns_false_when_only_safetensors_exists(self, tmp_path):
        (tmp_path / "layer.safetensors").touch()
        assert not is_layer_split(tmp_path, "layer")

    def test_returns_false_when_only_done_marker_exists(self, tmp_path):
        (tmp_path / "layer.safetensors.done").touch()
        assert not is_layer_split(tmp_path, "layer")

    def test_returns_true_when_both_exist(self, tmp_path):
        (tmp_path / "layer.safetensors").touch()
        (tmp_path / "layer.safetensors.done").touch()
        assert is_layer_split(tmp_path, "layer")


class TestSplitAndSaveLayers:
    def _create_single_file_model(
        self, model_path: Path, tensors: dict[str, torch.Tensor]
    ):
        """Create a model directory with a single model.safetensors file."""
        model_path.mkdir(parents=True, exist_ok=True)
        save_safetensors(tensors, model_path / "model.safetensors")

    def _create_sharded_model(
        self,
        model_path: Path,
        shard_contents: dict[str, dict[str, torch.Tensor]],
    ):
        """Create a model directory with multiple shards and an index file."""
        model_path.mkdir(parents=True, exist_ok=True)
        weight_map = {}
        for shard_name, tensors in shard_contents.items():
            save_safetensors(tensors, model_path / shard_name)
            for key in tensors:
                weight_map[key] = shard_name
        index = {"weight_map": weight_map}
        (model_path / "model.safetensors.index.json").write_text(json.dumps(index))

    def test_single_file_split(self, tmp_path):
        model_path = tmp_path / "model"
        tensors = {
            "model.embed_tokens.weight": torch.randn(10, 4),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(4, 4),
            "model.norm.weight": torch.randn(4),
            "lm_head.weight": torch.randn(10, 4),
        }
        self._create_single_file_model(model_path, tensors)

        layer_names = [
            "model.embed_tokens",
            "model.layers.0",
            "model.norm",
            "lm_head",
        ]
        split_dir = split_and_save_layers(model_path, layer_names)

        assert split_dir == model_path / SPLIT_DIR_NAME
        for name in layer_names:
            assert is_layer_split(split_dir, name)

    def test_single_file_split_preserves_weights(self, tmp_path):
        model_path = tmp_path / "model"
        embed_weight = torch.randn(10, 4)
        tensors = {
            "model.embed_tokens.weight": embed_weight.clone(),
            "model.norm.weight": torch.randn(4),
        }
        self._create_single_file_model(model_path, tensors)

        layer_names = ["model.embed_tokens", "model.norm"]
        split_dir = split_and_save_layers(model_path, layer_names)

        from chiquito.utils import load_safetensors

        loaded = load_safetensors(layer_file_path(split_dir, "model.embed_tokens"))
        torch.testing.assert_close(loaded["model.embed_tokens.weight"], embed_weight)

    def test_sharded_model_split(self, tmp_path):
        model_path = tmp_path / "model"
        shard_contents = {
            "model-00001-of-00002.safetensors": {
                "model.embed_tokens.weight": torch.randn(10, 4),
                "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
            },
            "model-00002-of-00002.safetensors": {
                "model.norm.weight": torch.randn(4),
                "lm_head.weight": torch.randn(10, 4),
            },
        }
        self._create_sharded_model(model_path, shard_contents)

        layer_names = [
            "model.embed_tokens",
            "model.layers.0",
            "model.norm",
            "lm_head",
        ]
        split_dir = split_and_save_layers(model_path, layer_names)

        assert split_dir == model_path / SPLIT_DIR_NAME
        for name in layer_names:
            assert is_layer_split(split_dir, name)

    def test_skips_already_split_layers(self, tmp_path):
        model_path = tmp_path / "model"
        tensors = {
            "model.embed_tokens.weight": torch.randn(10, 4),
            "model.norm.weight": torch.randn(4),
        }
        self._create_single_file_model(model_path, tensors)

        layer_names = ["model.embed_tokens", "model.norm"]

        # Split once
        split_dir = split_and_save_layers(model_path, layer_names)
        mtime_embed = layer_file_path(split_dir, "model.embed_tokens").stat().st_mtime

        # Split again - should skip
        split_dir2 = split_and_save_layers(model_path, layer_names)
        mtime_embed2 = layer_file_path(split_dir2, "model.embed_tokens").stat().st_mtime

        assert mtime_embed == mtime_embed2

    def test_quantize_skips_non_decoder_layers(self, tmp_path):
        """Non-decoder layers (embed, norm, lm_head) must stay fp16 in quantized splits."""
        model_path = tmp_path / "model"
        tensors = {
            "model.embed_tokens.weight": torch.randn(10, 4),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
            "model.norm.weight": torch.randn(4),
            "lm_head.weight": torch.randn(10, 4),
        }
        self._create_single_file_model(model_path, tensors)

        layer_names = [
            "model.embed_tokens",
            "model.layers.0",
            "model.norm",
            "lm_head",
        ]
        # Create base split first
        split_and_save_layers(model_path, layer_names)

        # Mock _quantize_state_dict to track which layers it's called on
        quantize_calls: list[dict] = []
        original_quantize = __import__(
            "chiquito.splitter", fromlist=["_quantize_state_dict"]
        )._quantize_state_dict

        def tracking_quantize(sd, quant, **kw):
            quantize_calls.append(dict(sd))
            return original_quantize(sd, quant, **kw)

        with patch("chiquito.splitter._quantize_state_dict", side_effect=tracking_quantize):
            split_and_save_layers(model_path, layer_names, quantization="4bit")

        # Only decoder layers should have been quantized
        assert len(quantize_calls) == 1
        assert "model.layers.0.self_attn.q_proj.weight" in quantize_calls[0]

    def test_quantize_skips_non_decoder_layers_composite_names(self, tmp_path):
        """Positional decoder detection works with composite model prefixes."""
        model_path = tmp_path / "model"
        tensors = {
            "model.language_model.embed_tokens.weight": torch.randn(10, 4),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
            "model.language_model.norm.weight": torch.randn(4),
            "lm_head.weight": torch.randn(10, 4),
        }
        self._create_single_file_model(model_path, tensors)

        layer_names = [
            "model.language_model.embed_tokens",
            "model.language_model.layers.0",
            "model.language_model.norm",
            "lm_head",
        ]
        split_and_save_layers(model_path, layer_names)

        quantize_calls: list[dict] = []
        original_quantize = __import__(
            "chiquito.splitter", fromlist=["_quantize_state_dict"]
        )._quantize_state_dict

        def tracking_quantize(sd, quant, **kw):
            quantize_calls.append(dict(sd))
            return original_quantize(sd, quant, **kw)

        with patch("chiquito.splitter._quantize_state_dict", side_effect=tracking_quantize):
            split_and_save_layers(model_path, layer_names, quantization="4bit")

        assert len(quantize_calls) == 1
        assert (
            "model.language_model.layers.0.self_attn.q_proj.weight"
            in quantize_calls[0]
        )

    def test_raises_when_no_model_files(self, tmp_path):
        import pytest

        model_path = tmp_path / "empty_model"
        model_path.mkdir()
        with pytest.raises(FileNotFoundError):
            split_and_save_layers(model_path, ["model.embed_tokens"])


class TestFindOrCreateSplit:
    @patch("chiquito.splitter.resolve_model_path")
    @patch("chiquito.splitter.split_and_save_layers")
    def test_returns_model_path_and_split_dir(self, mock_split, mock_resolve, tmp_path):
        model_path = tmp_path / "model"
        model_path.mkdir()
        split_dir = model_path / SPLIT_DIR_NAME
        split_dir.mkdir()

        mock_resolve.return_value = model_path
        mock_split.return_value = split_dir

        result_model_path, result_split_dir = find_or_create_split(
            str(model_path), ["layer1"]
        )

        assert result_model_path == model_path
        assert result_split_dir == split_dir

    @patch("chiquito.splitter.resolve_model_path")
    @patch("chiquito.splitter.split_and_save_layers")
    def test_passes_hf_token(self, mock_split, mock_resolve, tmp_path):
        model_path = tmp_path / "model"
        model_path.mkdir()
        mock_resolve.return_value = model_path
        mock_split.return_value = model_path / SPLIT_DIR_NAME

        find_or_create_split(str(model_path), ["layer1"], hf_token="tok")

        mock_split.assert_called_once_with(
            model_path,
            ["layer1"],
            hf_token="tok",
            repo_id=None,
            quantization=None,
            quantizable_params=None,
        )

    @patch("chiquito.splitter.resolve_model_path")
    @patch("chiquito.splitter.split_and_save_layers")
    def test_uses_repo_id_for_non_local_path(self, mock_split, mock_resolve, tmp_path):
        mock_resolve.return_value = tmp_path / "cached"
        mock_split.return_value = tmp_path / "cached" / SPLIT_DIR_NAME

        find_or_create_split("org/model", ["layer1"])

        mock_split.assert_called_once_with(
            tmp_path / "cached",
            ["layer1"],
            hf_token=None,
            repo_id="org/model",
            quantization=None,
            quantizable_params=None,
        )

    @patch("chiquito.splitter.resolve_model_path")
    @patch("chiquito.splitter.split_and_save_layers")
    def test_passes_quantization(self, mock_split, mock_resolve, tmp_path):
        model_path = tmp_path / "model"
        model_path.mkdir()
        mock_resolve.return_value = model_path
        mock_split.return_value = model_path / "chiquito_split_4bit"

        find_or_create_split(
            str(model_path), ["layer1"], quantization="4bit"
        )

        mock_split.assert_called_once_with(
            model_path,
            ["layer1"],
            hf_token=None,
            repo_id=None,
            quantization="4bit",
            quantizable_params=None,
        )


class TestSplitDirName:
    def test_no_quantization(self):
        assert split_dir_name(None) == SPLIT_DIR_NAME

    def test_4bit(self):
        assert split_dir_name("4bit") == f"{SPLIT_DIR_NAME}_4bit"

    def test_8bit(self):
        assert split_dir_name("8bit") == f"{SPLIT_DIR_NAME}_8bit"


class TestParseQuantizedStateDict:
    def test_plain_tensors_returned_as_base(self):
        raw = {
            "layer.weight": torch.randn(4, 4),
            "layer.bias": torch.randn(4),
        }
        base, qs = parse_quantized_state_dict(raw)
        assert set(base.keys()) == {"layer.weight", "layer.bias"}
        assert qs == {}

    def test_quant_state_entries_grouped(self):
        raw = {
            "layer.weight": torch.randint(0, 255, (8,), dtype=torch.uint8),
            "layer.weight.absmax": torch.randn(2),
            "layer.weight.quant_map": torch.randn(16),
            "layer.weight.quant_state.bitsandbytes__nf4": torch.tensor([]),
            "layer.bias": torch.randn(4),
        }
        base, qs = parse_quantized_state_dict(raw)

        assert set(base.keys()) == {"layer.weight", "layer.bias"}
        assert "layer.weight" in qs
        assert set(qs["layer.weight"].keys()) == {
            "absmax",
            "quant_map",
            "quant_state.bitsandbytes__nf4",
        }

    def test_nested_quant_entries(self):
        raw = {
            "w": torch.randint(0, 255, (8,), dtype=torch.uint8),
            "w.absmax": torch.randn(2),
            "w.nested_absmax": torch.randn(1),
            "w.nested_quant_map": torch.randn(16),
        }
        base, qs = parse_quantized_state_dict(raw)
        assert set(base.keys()) == {"w"}
        assert set(qs["w"].keys()) == {
            "absmax",
            "nested_absmax",
            "nested_quant_map",
        }

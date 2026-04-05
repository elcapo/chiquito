import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from transformers.cache_utils import DynamicCache

from chiquito.model import _SlidingWindowCache
from chiquito.utils import save_safetensors


def _make_layer_file(split_dir: Path, name: str, tensors: dict[str, torch.Tensor]):
    """Helper to create a layer safetensors file for testing."""
    split_dir.mkdir(parents=True, exist_ok=True)
    save_safetensors(tensors, split_dir / f"{name}.safetensors")


class TestSlidingWindowCache:
    def test_get_preloaded_layer(self, tmp_path):
        names = ["layer0", "layer1"]
        for n in names:
            _make_layer_file(tmp_path, n, {f"{n}.weight": torch.randn(2, 2)})

        cache = _SlidingWindowCache(names, tmp_path, window_size=2)
        cache.start()

        data = cache.get("layer0")
        assert "layer0.weight" in data
        cache.stop()

    def test_release_frees_layer(self, tmp_path):
        names = ["layer0", "layer1", "layer2"]
        for n in names:
            _make_layer_file(tmp_path, n, {f"{n}.weight": torch.randn(2, 2)})

        cache = _SlidingWindowCache(names, tmp_path, window_size=2)
        cache.start()

        # Get and release layer0
        cache.get("layer0")
        cache.release("layer0")

        # layer2 should eventually become available (background loader fills the gap)
        data = cache.get("layer2")
        assert "layer2.weight" in data
        cache.stop()

    def test_sequential_consumption(self, tmp_path):
        """Simulate a full forward pass consuming layers sequentially."""
        names = [f"layer{i}" for i in range(5)]
        for n in names:
            _make_layer_file(tmp_path, n, {f"{n}.weight": torch.randn(2, 2)})

        cache = _SlidingWindowCache(names, tmp_path, window_size=2)
        cache.start()

        for n in names:
            data = cache.get(n)
            assert f"{n}.weight" in data
            cache.release(n)

        cache.stop()

    def test_stop_is_idempotent(self, tmp_path):
        names = ["layer0"]
        _make_layer_file(tmp_path, "layer0", {"layer0.weight": torch.randn(2, 2)})

        cache = _SlidingWindowCache(names, tmp_path, window_size=1)
        cache.start()
        cache.stop()
        cache.stop()  # Should not raise

    def test_window_size_one(self, tmp_path):
        """Window of 1 means only one layer in memory at a time."""
        names = ["a", "b", "c"]
        for n in names:
            _make_layer_file(tmp_path, n, {f"{n}.w": torch.randn(2)})

        cache = _SlidingWindowCache(names, tmp_path, window_size=1)
        cache.start()

        for n in names:
            data = cache.get(n)
            assert f"{n}.w" in data
            cache.release(n)

        cache.stop()


class TestPrepareInputsForGeneration:
    """Test prepare_inputs_for_generation without instantiating ChiquitoModel."""

    def _call_prepare(self, input_ids, attention_mask=None, **kwargs):
        """Call the method unbound, using a minimal mock self."""
        from chiquito.model import ChiquitoModel

        return ChiquitoModel.prepare_inputs_for_generation(
            None, input_ids, attention_mask=attention_mask, **kwargs
        )

    def test_no_past_key_values(self):
        input_ids = torch.tensor([[1, 2, 3]])
        result = self._call_prepare(input_ids)
        assert result["input_ids"] is input_ids
        assert result["past_key_values"] is None
        assert result["use_cache"] is True

    def test_with_past_key_values_trims_input(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        past_kv = MagicMock(spec=DynamicCache)
        past_kv.get_seq_length.return_value = 3

        result = self._call_prepare(input_ids, past_key_values=past_kv)

        # Should trim to tokens after past_len
        assert result["input_ids"].shape[1] == 2
        assert result["input_ids"].tolist() == [[4, 5]]

    def test_with_past_key_values_last_token_only(self):
        input_ids = torch.tensor([[1, 2, 3]])
        past_kv = MagicMock(spec=DynamicCache)
        past_kv.get_seq_length.return_value = 5  # past_len > seq_len

        result = self._call_prepare(input_ids, past_key_values=past_kv)
        assert result["input_ids"].shape[1] == 1
        assert result["input_ids"].tolist() == [[3]]

    def test_position_ids_from_attention_mask(self):
        input_ids = torch.tensor([[10, 20, 30]])
        attention_mask = torch.tensor([[1, 1, 1]])

        result = self._call_prepare(input_ids, attention_mask=attention_mask)

        assert result["position_ids"].tolist() == [[0, 1, 2]]

    def test_position_ids_with_padding(self):
        input_ids = torch.tensor([[0, 10, 20]])
        attention_mask = torch.tensor([[0, 1, 1]])

        result = self._call_prepare(input_ids, attention_mask=attention_mask)

        # Padded positions get position_id 1 (masked fill), real tokens get cumsum
        assert result["position_ids"].tolist() == [[1, 0, 1]]

    def test_position_ids_trimmed_with_past(self):
        input_ids = torch.tensor([[1, 2, 3, 4]])
        attention_mask = torch.tensor([[1, 1, 1, 1]])
        past_kv = MagicMock(spec=DynamicCache)
        past_kv.get_seq_length.return_value = 2

        result = self._call_prepare(
            input_ids, attention_mask=attention_mask, past_key_values=past_kv
        )

        # input_ids trimmed to last 2, position_ids trimmed to last 2
        assert result["input_ids"].tolist() == [[3, 4]]
        assert result["position_ids"].shape[1] == 2


class TestChiquitoModelProperties:
    """Test simple properties/methods that don't need full model init."""

    def test_can_generate(self):
        from chiquito.model import ChiquitoModel

        assert ChiquitoModel.can_generate(None) is True

    def test_layer_names_class_var(self):
        from chiquito.model import ChiquitoModel

        names = ChiquitoModel.LAYER_NAMES
        assert names["embed"] == "model.embed_tokens"
        assert names["layer_prefix"] == "model.layers"
        assert names["norm"] == "model.norm"
        assert names["lm_head"] == "lm_head"

    def test_call_delegates_to_forward(self):
        from chiquito.model import ChiquitoModel

        mock_self = MagicMock(spec=ChiquitoModel)
        mock_self.forward = MagicMock(return_value="result")
        mock_self.__call__ = ChiquitoModel.__call__

        result = mock_self.__call__(mock_self, input_ids=torch.tensor([[1]]))
        mock_self.forward.assert_called_once()

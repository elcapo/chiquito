from pathlib import Path
from unittest.mock import patch, MagicMock

import torch

from chiquito.utils import (
    clean_gpu_memory,
    clean_memory,
    load_safetensors,
    resolve_model_path,
    save_safetensors,
)


class TestCleanMemory:
    def test_clean_memory_runs_without_error(self):
        clean_memory()

    def test_clean_gpu_memory_runs_without_error(self):
        clean_gpu_memory()

    @patch("chiquito.utils.torch.cuda.is_available", return_value=True)
    @patch("chiquito.utils.torch.cuda.empty_cache")
    def test_clean_gpu_memory_calls_empty_cache_when_cuda_available(
        self, mock_empty_cache, mock_is_available
    ):
        clean_gpu_memory()
        mock_empty_cache.assert_called_once()

    @patch("chiquito.utils.torch.cuda.is_available", return_value=False)
    @patch("chiquito.utils.torch.cuda.empty_cache")
    def test_clean_gpu_memory_skips_when_no_cuda(
        self, mock_empty_cache, mock_is_available
    ):
        clean_gpu_memory()
        mock_empty_cache.assert_not_called()

    @patch("chiquito.utils.torch.cuda.is_available", return_value=True)
    @patch("chiquito.utils.torch.cuda.empty_cache")
    def test_clean_memory_calls_empty_cache_when_cuda_available(
        self, mock_empty_cache, mock_is_available
    ):
        clean_memory()
        mock_empty_cache.assert_called_once()


class TestSafetensorsRoundTrip:
    def test_save_and_load(self, tmp_path):
        state_dict = {
            "weight": torch.randn(4, 4),
            "bias": torch.randn(4),
        }
        path = tmp_path / "test.safetensors"
        save_safetensors(state_dict, path)

        loaded = load_safetensors(path)

        assert set(loaded.keys()) == {"weight", "bias"}
        torch.testing.assert_close(loaded["weight"], state_dict["weight"])
        torch.testing.assert_close(loaded["bias"], state_dict["bias"])

    def test_load_returns_cpu_tensors(self, tmp_path):
        state_dict = {"x": torch.randn(2, 2)}
        path = tmp_path / "test.safetensors"
        save_safetensors(state_dict, path)

        loaded = load_safetensors(path)
        assert loaded["x"].device == torch.device("cpu")

    def test_multiple_tensors_round_trip(self, tmp_path):
        state_dict = {
            f"layer.{i}": torch.randn(3, 3) for i in range(5)
        }
        path = tmp_path / "multi.safetensors"
        save_safetensors(state_dict, path)

        loaded = load_safetensors(path)
        assert len(loaded) == 5
        for k in state_dict:
            torch.testing.assert_close(loaded[k], state_dict[k])


class TestResolveModelPath:
    def test_local_dir_with_index_json(self, tmp_path):
        (tmp_path / "model.safetensors.index.json").touch()
        result = resolve_model_path(str(tmp_path))
        assert result == tmp_path

    def test_local_dir_with_single_safetensors(self, tmp_path):
        (tmp_path / "model.safetensors").touch()
        result = resolve_model_path(str(tmp_path))
        assert result == tmp_path

    @patch("chiquito.utils.huggingface_hub.snapshot_download")
    def test_local_dir_without_model_files_falls_through_to_hf(
        self, mock_download, tmp_path
    ):
        mock_download.return_value = str(tmp_path / "cached")
        result = resolve_model_path(str(tmp_path))
        mock_download.assert_called_once()
        assert result == tmp_path / "cached"

    @patch("chiquito.utils.huggingface_hub.snapshot_download")
    def test_hf_repo_id(self, mock_download, tmp_path):
        cached = tmp_path / "cached_model"
        cached.mkdir()
        mock_download.return_value = str(cached)

        result = resolve_model_path("org/model-name")

        mock_download.assert_called_once_with(
            "org/model-name", token=None, ignore_patterns=["*.bin"]
        )
        assert result == cached

    @patch("chiquito.utils.huggingface_hub.snapshot_download")
    def test_hf_repo_id_with_token(self, mock_download, tmp_path):
        cached = tmp_path / "cached_model"
        cached.mkdir()
        mock_download.return_value = str(cached)

        resolve_model_path("org/model-name", hf_token="hf_test123")

        mock_download.assert_called_once_with(
            "org/model-name", token="hf_test123", ignore_patterns=["*.bin"]
        )

from unittest.mock import MagicMock, patch

import pytest

from chiquito.auto_model import AutoModel


class TestAutoModelInit:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(OSError, match="AutoModel is designed to be instantiated"):
            AutoModel()


class TestAutoModelRegistry:
    def setup_method(self):
        self._original_registry = AutoModel._REGISTRY.copy()

    def teardown_method(self):
        AutoModel._REGISTRY = self._original_registry

    def test_register_and_lookup(self):
        mock_cls = MagicMock()
        AutoModel.register("TestArch", mock_cls)
        assert "TestArch" in AutoModel._REGISTRY
        assert AutoModel._REGISTRY["TestArch"] is mock_cls

    def test_register_overwrites(self):
        mock_cls1 = MagicMock()
        mock_cls2 = MagicMock()
        AutoModel.register("Arch", mock_cls1)
        AutoModel.register("Arch", mock_cls2)
        assert AutoModel._REGISTRY["Arch"] is mock_cls2


class TestAutoModelFromPretrained:
    def setup_method(self):
        self._original_registry = AutoModel._REGISTRY.copy()

    def teardown_method(self):
        AutoModel._REGISTRY = self._original_registry

    @patch("chiquito.auto_model.AutoConfig.from_pretrained")
    @patch("chiquito.model.ChiquitoModel.__init__", return_value=None)
    def test_falls_back_to_chiquito_model(self, mock_init, mock_config):
        config = MagicMock()
        config.architectures = ["UnknownArchitecture"]
        mock_config.return_value = config

        AutoModel._REGISTRY.clear()

        result = AutoModel.from_pretrained("some/model")
        mock_init.assert_called_once_with("some/model")

    @patch("chiquito.auto_model.AutoConfig.from_pretrained")
    def test_uses_registered_class_when_arch_matches(self, mock_config):
        config = MagicMock()
        config.architectures = ["LlamaForCausalLM"]
        mock_config.return_value = config

        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        AutoModel.register("Llama", mock_cls)

        result = AutoModel.from_pretrained("meta/llama", device="cpu")

        mock_cls.assert_called_once_with("meta/llama", device="cpu")
        assert result is mock_instance

    @patch("chiquito.auto_model.AutoConfig.from_pretrained")
    @patch("chiquito.model.ChiquitoModel.__init__", return_value=None)
    def test_passes_hf_token_to_config(self, mock_init, mock_config):
        config = MagicMock()
        config.architectures = None
        mock_config.return_value = config

        AutoModel.from_pretrained("some/model", hf_token="tok123")

        mock_config.assert_called_once_with(
            "some/model", trust_remote_code=True, token="tok123"
        )

    @patch("chiquito.auto_model.AutoConfig.from_pretrained")
    @patch("chiquito.model.ChiquitoModel.__init__", return_value=None)
    def test_no_architectures_falls_back(self, mock_init, mock_config):
        config = MagicMock()
        config.architectures = None
        mock_config.return_value = config

        mock_cls = MagicMock()
        AutoModel.register("SomeArch", mock_cls)

        AutoModel.from_pretrained("some/model")
        mock_init.assert_called_once()
        mock_cls.assert_not_called()

    @patch("chiquito.auto_model.AutoConfig.from_pretrained")
    def test_partial_arch_match(self, mock_config):
        """Registry key 'Llama' should match architecture 'LlamaForCausalLM'."""
        config = MagicMock()
        config.architectures = ["LlamaForCausalLM"]
        mock_config.return_value = config

        mock_cls = MagicMock()
        mock_cls.return_value = MagicMock()
        AutoModel.register("Llama", mock_cls)

        result = AutoModel.from_pretrained("org/model")
        mock_cls.assert_called_once()

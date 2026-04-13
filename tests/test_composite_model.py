from unittest.mock import MagicMock

import torch

from chiquito.composite_model import ChiquitoCompositeModel


class TestRemapStateDictKeys:
    def test_remaps_language_model_prefix(self):
        state_dict = {
            "model.language_model.embed_tokens.weight": torch.randn(4, 4),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(4, 4),
        }
        model = object.__new__(ChiquitoCompositeModel)
        result = model._remap_state_dict_keys(state_dict)

        assert "model.embed_tokens.weight" in result
        assert "model.layers.0.self_attn.q_proj.weight" in result
        assert "model.language_model.embed_tokens.weight" not in result

    def test_leaves_lm_head_unchanged(self):
        state_dict = {"lm_head.weight": torch.randn(4, 4)}
        model = object.__new__(ChiquitoCompositeModel)
        result = model._remap_state_dict_keys(state_dict)

        assert "lm_head.weight" in result

    def test_preserves_tensor_identity(self):
        t = torch.randn(2, 2)
        state_dict = {"model.language_model.norm.weight": t}
        model = object.__new__(ChiquitoCompositeModel)
        result = model._remap_state_dict_keys(state_dict)

        assert result["model.norm.weight"] is t


class TestBaseRemapIsNoop:
    def test_base_class_returns_same_dict(self):
        from chiquito.model import ChiquitoModel

        state_dict = {"model.layers.0.weight": torch.randn(2, 2)}
        model = object.__new__(ChiquitoModel)
        result = model._remap_state_dict_keys(state_dict)

        assert result is state_dict


class TestCompositeLayerNames:
    def test_disk_layer_names_use_language_model_prefix(self):
        names = ChiquitoCompositeModel.LAYER_NAMES
        assert "language_model" in names["embed"]
        assert "language_model" in names["layer_prefix"]
        assert "language_model" in names["norm"]
        assert "language_model" not in names["lm_head"]

    def test_model_layer_names_match_base_class(self):
        from chiquito.model import ChiquitoModel

        model_names = ChiquitoCompositeModel._MODEL_LAYER_NAMES
        base_names = ChiquitoModel.LAYER_NAMES
        assert model_names == base_names


class TestCompositeBuildLayers:
    def test_build_layers_uses_model_paths_for_traversal(self):
        """_build_layers should traverse model-object paths but store disk names."""
        model = object.__new__(ChiquitoCompositeModel)

        # Create a mock model object with the model-object structure
        mock_embed = MagicMock()
        mock_layer0 = MagicMock()
        mock_layer1 = MagicMock()
        mock_norm = MagicMock()
        mock_lm_head = MagicMock()

        # Build a mock with the right attribute structure
        mock_model = MagicMock()
        mock_model.model.embed_tokens = mock_embed
        # Use a MagicMock that behaves like a list for iteration
        mock_layers_container = MagicMock()
        mock_layers_container.__iter__ = MagicMock(
            return_value=iter([mock_layer0, mock_layer1])
        )
        mock_model.model.layers = mock_layers_container
        mock_model.model.norm = mock_norm
        mock_model.lm_head = mock_lm_head
        model.model = mock_model

        model._build_layers()

        # layer_names should use disk format
        assert model.layer_names[0] == "model.language_model.embed_tokens"
        assert model.layer_names[1] == "model.language_model.layers.0"
        assert model.layer_names[2] == "model.language_model.layers.1"
        assert model.layer_names[3] == "model.language_model.norm"
        assert model.layer_names[4] == "lm_head"

        # layers should reference the model-object modules
        assert model.layers[0] is mock_embed
        assert model.layers[1] is mock_layer0
        assert model.layers[2] is mock_layer1
        assert model.layers[3] is mock_norm
        assert model.layers[4] is mock_lm_head


class TestAutoModelRegistration:
    def test_qwen3_5_moe_registered(self):
        from chiquito.auto_model import AutoModel

        assert "Qwen3_5Moe" in AutoModel._REGISTRY
        assert AutoModel._REGISTRY["Qwen3_5Moe"] is ChiquitoCompositeModel

from .auto_model import AutoModel
from .composite_model import ChiquitoCompositeModel
from .model import ChiquitoModel

AutoModel.register("Qwen3_5Moe", ChiquitoCompositeModel)

__all__ = ["AutoModel", "ChiquitoCompositeModel", "ChiquitoModel"]

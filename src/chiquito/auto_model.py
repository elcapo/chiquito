from __future__ import annotations

from typing import TYPE_CHECKING

from transformers import AutoConfig

if TYPE_CHECKING:
    from .model import ChiquitoModel


class AutoModel:
    _REGISTRY: dict[str, type[ChiquitoModel]] = {}

    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated using "
            "AutoModel.from_pretrained(pretrained_model_name_or_path)"
        )

    @classmethod
    def register(cls, arch_name: str, model_class: type[ChiquitoModel]):
        cls._REGISTRY[arch_name] = model_class

    @classmethod
    def from_pretrained(cls, model_id_or_path: str, **kwargs) -> ChiquitoModel:
        from .model import ChiquitoModel

        hf_token = kwargs.get("hf_token")
        token_kwargs = {"token": hf_token} if hf_token else {}
        config = AutoConfig.from_pretrained(
            model_id_or_path, trust_remote_code=True, **token_kwargs
        )

        arch = config.architectures[0] if getattr(config, "architectures", None) else ""

        for key, model_cls in cls._REGISTRY.items():
            if key in arch:
                return model_cls(model_id_or_path, **kwargs)

        return ChiquitoModel(model_id_or_path, **kwargs)

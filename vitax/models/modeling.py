from typing import Union, Dict, Any, List

from .vit import VisionTransformer

from .vit_load_weights import vit_get_model , VisionTransformers


class ViTFactory:

    def can_build(self, name_or_config: Union[str, Dict]) -> bool:

        if isinstance(name_or_config, dict):
            return True

        if isinstance(name_or_config, str) and name_or_config in VisionTransformers:
            return True
        return False


    def build(self, name_or_config: Union[str, Dict], **kwargs: Any) -> VisionTransformer:

        return vit_get_model(name_or_config=name_or_config, **kwargs)


_MODEL_FACTORIES: List[Any] = [
    ViTFactory(),
]


# --- The Main Public API Function ---

def create_model(
    name_or_config: Union[str, Dict],
    **kwargs: Any,
) -> VisionTransformer:

    for factory in _MODEL_FACTORIES:

        if factory.can_build(name_or_config):

            return factory.build(name_or_config, **kwargs)

    raise ValueError(f"Could not find a model factory for '{name_or_config}'.")
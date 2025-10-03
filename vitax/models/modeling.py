from typing import Union, Dict, Any
from .vit import VisionTransformer
from .vit_load_weights import get_model as get_vit_from_hf, VisionTransformers


_MODEL_REGISTRY = {}

def register_model(name: str):
    """A decorator to register a new model builder function."""

    def decorator(f):
        _MODEL_REGISTRY[name] = f
        return f

    return decorator


@register_model("vit")
def _build_vit(name_or_config: Union[str, Dict], pretrained: bool = False, **kwargs: Any) -> VisionTransformer:

    if pretrained:
        if not isinstance(name_or_config, str):
            raise ValueError("To load pretrained weights, 'name_or_config' must be a model name string.")
        if name_or_config not in VisionTransformers:
            raise ValueError(f"Model '{name_or_config}' is not a supported pretrained Vision Transformer.")

        # We can directly use the robust `get_model` function we already wrote
        return get_vit_from_hf(name_or_config=name_or_config, pretrained=True, **kwargs)

    # Case 2: Create a model from a standard config name (but with random weights)
    if isinstance(name_or_config, str):
        return get_vit_from_hf(name_or_config=name_or_config, pretrained=False, **kwargs)

    # Case 3: Create a model from a custom dictionary config
    if isinstance(name_or_config, dict):
        # Combine the config dict with other kwargs like num_classes
        model_kwargs = {**name_or_config, **kwargs}
        return VisionTransformer(**model_kwargs)

    raise TypeError(f"Unsupported type for 'name_or_config': {type(name_or_config)}")


def create_model(model_name: str, **kwargs: Any) -> VisionTransformer:

    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: '{model_name}'. Available models: {list(_MODEL_REGISTRY.keys())}")

    # Look up the correct builder function in the registry and call it
    builder_fn = _MODEL_REGISTRY[model_name]
    return builder_fn(**kwargs)
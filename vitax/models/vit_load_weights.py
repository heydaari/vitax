from transformers import FlaxViTForImageClassification
from transformers.models.vit.configuration_vit import ViTConfig
from flax import nnx
from typing import Union
from vitax.models.vit import VisionTransformer

VisionTransformers = ['google/vit-base-patch16-224',
                    'google/vit-large-patch16-224',
                    'google/vit-large-patch16-384',
                    'google/vit-base-patch16-384',
                    'google/vit-base-patch32-384',
                    'google/vit-large-patch32-384',
                    'google/vit-large-patch16-224-in21k',
                    'google/vit-base-patch16-224-in21k',
                    'google/vit-huge-patch14-224-in21k',
                    'google/vit-base-patch32-224-in21k',
                    'google/vit-large-patch32-224-in21k']

def vit_inplace_copy_weights(*, src_model, dst_model, config):

    assert isinstance(src_model, FlaxViTForImageClassification)
    assert isinstance(dst_model, VisionTransformer)

    tf_model_params = src_model.params
    tf_model_params_fstate = nnx.traversals.flatten_mapping(tf_model_params)

    flax_model_params = nnx.state(dst_model, nnx.Param)
    flax_model_params_fstate = dict(flax_model_params.flat_state())

    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // num_heads

    params_name_mapping = {
        ("cls_token",): ("vit", "embeddings", "cls_token"),
        ("position_embeddings",): ("vit", "embeddings", "position_embeddings"),
        **{
            ("patch_embeddings", x): ("vit", "embeddings", "patch_embeddings", "projection", x)
            for x in ["kernel", "bias"]
        },
        **{
            ("encoder", "layers", i, "attn", y, x): (
                "vit", "encoder", "layer", str(i), "attention", "attention", y, x
            )
            for x in ["kernel", "bias"]
            for y in ["key", "value", "query"]
            for i in range(num_layers)
        },
        **{
            ("encoder", "layers", i, "attn", "out", x): (
                "vit", "encoder", "layer", str(i), "attention", "output", "dense", x
            )
            for x in ["kernel", "bias"]
            for i in range(num_layers)
        },
        **{
            ("encoder", "layers", i, "mlp", "layers", y1, x): (
                "vit", "encoder", "layer", str(i), y2, "dense", x
            )
            for x in ["kernel", "bias"]
            for y1, y2 in [(0, "intermediate"), (3, "output")]
            for i in range(num_layers)
        },
        **{
            ("encoder", "layers", i, y1, x): (
                "vit", "encoder", "layer", str(i), y2, x
            )
            for x in ["scale", "bias"]
            for y1, y2 in [("norm1", "layernorm_before"), ("norm2", "layernorm_after")]
            for i in range(num_layers)
        },
        **{
            ("final_norm", x): ("vit", "layernorm", x)
            for x in ["scale", "bias"]
        },
        **{
            ("classifier", x): ("classifier", x)
            for x in ["kernel", "bias"]
        }
    }

    nonvisited = set(flax_model_params_fstate.keys())

    for key1, key2 in params_name_mapping.items():
        assert key1 in flax_model_params_fstate, key1
        assert key2 in tf_model_params_fstate, (key1, key2)

        nonvisited.remove(key1)

        src_value = tf_model_params_fstate[key2]
        if key2[-1] == "kernel" and key2[-2] in ("key", "value", "query"):
            shape = src_value.shape
            src_value = src_value.reshape((shape[0], num_heads, head_dim))

        if key2[-1] == "bias" and key2[-2] in ("key", "value", "query"):
            src_value = src_value.reshape((num_heads, head_dim))

        if key2[-4:] == ("attention", "output", "dense", "kernel"):
            shape = src_value.shape
            src_value = src_value.reshape((num_heads, head_dim, shape[-1]))

        dst_value = flax_model_params_fstate[key1]
        assert src_value.shape == dst_value.value.shape, (key2, src_value.shape, key1, dst_value.value.shape)
        dst_value.value = src_value.copy()
        assert dst_value.value.mean() == src_value.mean(), (dst_value.value, src_value.mean())

    assert len(nonvisited) == 0, nonvisited

    nnx.update(dst_model, nnx.State.from_flat_path(flax_model_params_fstate))


def vit_get_model(
    name_or_config: Union[str, dict],
    num_classes: int = 1000,
    pretrained: bool = True
) -> VisionTransformer:

    if pretrained and isinstance(name_or_config, dict):
        raise ValueError("Loading pretrained weights requires a model name (string), not a config dictionary.")

    if isinstance(name_or_config, str) and name_or_config not in VisionTransformers:
        raise ValueError(f"Model name '{name_or_config}' is not a valid or available model.")

    if pretrained:

        hf_model = FlaxViTForImageClassification.from_pretrained(name_or_config)
        config = hf_model.config

        model = VisionTransformer(
            num_classes=config.num_labels,
            img_size=config.image_size,
            patch_size=config.patch_size,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            mlp_dim=config.intermediate_size,
            hidden_size=config.hidden_size,
            dropout_rate=config.hidden_dropout_prob,
            rngs=nnx.Rngs(0)
        )

        # Use the inplace function to copy the weights
        vit_inplace_copy_weights(src_model=hf_model, dst_model=model, config=config)

        # If the user specified a different number of classes for a fine-tuning task,
        # replace the classifier head *after* loading all other pre-trained weights.
        if num_classes != config.num_labels:
            model.classifier = nnx.Linear(
                in_features=model.classifier.in_features,
                out_features=num_classes,
                rngs=nnx.Rngs(0)
            )
    else:  # Not pretrained, create a randomly initialized model
        model_kwargs = {}
        if isinstance(name_or_config, str):
            # Get the default configuration for the named model
            config = ViTConfig.from_pretrained(name_or_config)
            model_kwargs = {
                'img_size': config.image_size,
                'patch_size': config.patch_size,
                'num_layers': config.num_hidden_layers,
                'num_heads': config.num_attention_heads,
                'mlp_dim': config.intermediate_size,
                'hidden_size': config.hidden_size,
                'dropout_rate': config.hidden_dropout_prob,
            }
        elif isinstance(name_or_config, dict):
            config_key_mapping = {
                'image_size': 'img_size',
                'patch_size': 'patch_size',
                'num_hidden_layers': 'num_layers',
                'num_attention_heads': 'num_heads',
                'intermediate_size': 'mlp_dim',
                'hidden_size': 'hidden_size',
                'hidden_dropout_prob': 'dropout_rate',
            }
            for key in name_or_config.keys():
                if key in config_key_mapping:
                    model_kwargs[config_key_mapping[key]] = name_or_config[key]
        else:
            raise TypeError(f"Expected 'name_or_config' to be a str or dict, but got {type(name_or_config)}.")

        model = VisionTransformer(
            num_classes=num_classes,
            **model_kwargs,
            rngs=nnx.Rngs(0)
        )

    return model
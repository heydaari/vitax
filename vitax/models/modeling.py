from typing import Union, Dict, Any, List
import jax
from .vit import VisionTransformer
from .vit_load_weights import vit_get_model , VisionTransformers
from flax import nnx
import optax

@nnx.jit(static_argnames=['mesh', 'optimizer'])
def shard_model_and_create_optimizer(model: VisionTransformer,
                                     optimizer, # Should be an Optax optimizer object
                                     mesh
                                     ):

    optimizer = nnx.optimizer.Optimizer(model=model, tx=optimizer, wrt=nnx.Param)

    model_state = nnx.state(model)
    model_shardings = nnx.get_named_sharding(model_state, mesh)
    model_sharded_state = jax.lax.with_sharding_constraint(model_state, model_shardings)
    nnx.update(model, model_sharded_state)

    optimizer_state = nnx.state(optimizer, nnx.optimizer.OptState)
    optimizer_shardings = nnx.get_named_sharding(optimizer_state, mesh)
    optimizer_sharded_state = jax.lax.with_sharding_constraint(optimizer_state, optimizer_shardings)
    nnx.update(optimizer, optimizer_sharded_state)

    return model, optimizer

class ViTFactory:

    def can_build(self,
                  name_or_config: Union[str, Dict]) -> bool:

        if isinstance(name_or_config, dict):
            return True

        if isinstance(name_or_config, str) and name_or_config in VisionTransformers:
            return True
        return False


    def build(self,
              name_or_config: Union[str, Dict],
              fsdp: bool,
              mesh=None,
              optimizer=None,
              **kwargs: Any):

        if fsdp:

            with mesh:
                model = vit_get_model(name_or_config=name_or_config, fsdp=fsdp, **kwargs)

            # This returns sharded model and optimizer, should be unpacked when calling the create_model function
            return shard_model_and_create_optimizer(model=model,
                                                    mesh=mesh,
                                                    optimizer=optimizer)
        else:

            model = vit_get_model(name_or_config=name_or_config, fsdp=fsdp, **kwargs)
            return model


_MODEL_FACTORIES: List[Any] = [
    ViTFactory(),
]


# --- The Main Public API Function ---

def create_model(
    name_or_config: Union[str, Dict],
    fsdp: bool = False,
    optimizer=None,
    **kwargs: Any,
) -> VisionTransformer:

    if fsdp and not optimizer:
        raise ValueError("For FSDP, you should provide an optax optimizer.")

    if fsdp:
        num_devices = jax.device_count()
        mesh = jax.make_mesh(
            axis_shapes=(num_devices,),
            axis_names=('data',),
        )
    else: 
        mesh = None

    for factory in _MODEL_FACTORIES:

        if factory.can_build(name_or_config):

            return factory.build(name_or_config=name_or_config,
                                 fsdp=fsdp,
                                 mesh=mesh,
                                 optimizer=optimizer,
                                 **kwargs)

    raise ValueError(f"Could not find a model factory for '{name_or_config}'.")
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import dtypes, initializers

default_kernel_init = initializers.lecun_normal()


class VisionTransformer(nnx.Module):
    """ Implements the ViT model, inheriting from `flax.nnx.Module`. """

    def __init__(
            self,
            num_classes: int = 1000,
            in_channels: int = 3,
            img_size: int = 224,
            patch_size: int = 16,
            num_layers: int = 12,
            num_heads: int = 12,
            mlp_dim: int = 3072,
            hidden_size: int = 768,
            dropout_rate: float = 0.1,
            fsdp: bool = True,
            *,
            rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        n_patches = (img_size // patch_size) ** 2

        # We shard the kernel of the convolution layer along the output feature dimension ('hidden_size').
        # This distributes the initial projection across devices.
        self.patch_embeddings = nnx.Conv(
            in_channels,
            hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            use_bias=True,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(),
                (None, None, None, 'data'),  # Shard along the output feature axis
            ) if fsdp else default_kernel_init,
            rngs=rngs,
        )

        initializer = jax.nn.initializers.truncated_normal(stddev=0.02)
        self.position_embeddings = nnx.Param(
            initializer(rngs.params(), (1, n_patches + 1, hidden_size), jnp.float32)
        )
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.cls_token = nnx.Param(jnp.zeros((1, 1, hidden_size)))

        self.encoder = nnx.Sequential(*[
            TransformerEncoder(hidden_size, mlp_dim, num_heads,
                               dropout_rate, fsdp=fsdp, rngs=rngs)
            for i in range(num_layers)
        ])
        self.final_norm = nnx.LayerNorm(hidden_size, rngs=rngs)

        # We shard the final classification layer's kernel along its input dimension.
        self.classifier = nnx.Linear(
            hidden_size,
            num_classes,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.lecun_normal(),
                ('data', None),  # Shard along the input feature axis
            ) if fsdp else default_kernel_init,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        patches = self.patch_embeddings(x)
        batch_size = patches.shape[0]
        patches = patches.reshape(batch_size, -1, patches.shape[-1])

        cls_token = jnp.tile(self.cls_token, [batch_size, 1, 1])
        x = jnp.concat([cls_token, patches], axis=1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)

        x = self.encoder(embeddings)
        x = self.final_norm(x)
        x = x[:, 0]

        return self.classifier(x)


class TransformerEncoder(nnx.Module):
    """ A single transformer encoder block in the ViT model. """

    def __init__(
            self,
            hidden_size: int,
            mlp_dim: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            fsdp:bool = True,
            *,
            rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:
        self.norm1 = nnx.LayerNorm(hidden_size, rngs=rngs)

        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_size,
            dropout_rate=dropout_rate,
            broadcast_dropout=False,
            decode=False,
            deterministic=False,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.xavier_uniform(),
                (None, 'data', None) ,  # Shard Q,K,V kernels
            ) if fsdp else default_kernel_init,

            out_kernel_init=nnx.with_partitioning(
                nnx.initializers.xavier_uniform(),
                ('data', None) if fsdp else default_kernel_init,  # Shard output projection
            ) if fsdp else default_kernel_init,
            rngs=rngs,
        )

        self.norm2 = nnx.LayerNorm(hidden_size, rngs=rngs)

        self.mlp = nnx.Sequential(
            nnx.Linear(
                hidden_size,
                mlp_dim,
                kernel_init=nnx.with_partitioning(
                    nnx.initializers.lecun_normal(), (None, 'data')
                ) if fsdp else default_kernel_init,
                rngs=rngs
            ),
            nnx.gelu,
            nnx.Dropout(dropout_rate, rngs=rngs),
            nnx.Linear(
                mlp_dim,
                hidden_size,  # Corrected output dimension
                kernel_init=nnx.with_partitioning(
                    nnx.initializers.lecun_normal(), ('data', None)
                ) if fsdp else default_kernel_init,
                rngs=rngs
            ),
            nnx.Dropout(dropout_rate, rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
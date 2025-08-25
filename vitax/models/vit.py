import jax
import jax.numpy as jnp
from flax import nnx


class VisionTransformer(nnx.Module):
    """ Implements the ViT model, inheriting from `flax.nnx.Module`.

    Args:
        num_classes (int): Number of classes in the classification. Defaults to 1000.
        in_channels (int): Number of input channels in the image (such as 3 for RGB). Defaults to 3.
        img_size (int): Input image size. Defaults to 224.
        patch_size (int): Size of the patches extracted from the image. Defaults to 16.
        num_layers (int): Number of transformer encoder layers. Defaults to 12.
        num_heads (int): Number of attention heads in each transformer layer. Defaults to 12.
        mlp_dim (int): Dimension of the hidden layers in the feed-forward/MLP block. Defaults to 3072.
        hidden_size (int): Dimensionality of the embedding vectors. Defaults to 3072.
        dropout_rate (int): Dropout rate (for regularization). Defaults to 0.1.
        rngs (flax.nnx.Rngs): A set of named `flax.nnx.RngStream` objects that generate a stream of JAX pseudo-random number generator (PRNG) keys. Defaults to `flax.nnx.Rngs(0)`.

    """

    def __init__(
            self,
            num_classes: int = 1000,
            in_channels: int = 3,
            img_size: int = 224,
            patch_size: int = 16,
            num_layers: int = 24,
            num_heads: int = 16,
            mlp_dim: int = 4096,
            hidden_size: int = 1024,
            dropout_rate: float = 0.1,
            *,
            rngs: nnx.Rngs = nnx.Rngs(0),
    ):

        n_patches = (img_size // patch_size) ** 2

        self.patch_embeddings = nnx.Conv(
            in_channels,
            hidden_size,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding="VALID",
            use_bias=True,
            rngs=rngs,
        )

        initializer = jax.nn.initializers.truncated_normal(stddev=0.02)

        self.position_embeddings = nnx.Param(
            initializer(rngs.params(), (1, n_patches + 1, hidden_size), jnp.float32)
        )

        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        self.cls_token = nnx.Param(jnp.zeros((1, 1, hidden_size)))

        self.encoder = nnx.Sequential(*[
            TransformerEncoder(hidden_size, mlp_dim, num_heads, dropout_rate, rngs=rngs)
            for i in range(num_layers)
        ])

        self.final_norm = nnx.LayerNorm(hidden_size, rngs=rngs)

        self.classifier = nnx.Linear(hidden_size, num_classes, rngs=rngs)

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

        # CLS Pooling
        x = x[:, 0]

        return self.classifier(x)


class TransformerEncoder(nnx.Module):
    """
    A single transformer encoder block in the ViT model, inheriting from `flax.nnx.Module`.

    Args:
        hidden_size (int): Input/output embedding dimensionality.
        mlp_dim (int): Dimension of the feed-forward/MLP block hidden layer.
        num_heads (int): Number of attention heads.
        dropout_rate (float): Dropout rate. Defaults to 0.0.
        rngs (flax.nnx.Rngs): A set of named `flax.nnx.RngStream` objects that generate a stream of JAX pseudo-random number generator (PRNG) keys. Defaults to `flax.nnx.Rngs(0)`.
    """

    def __init__(
            self,
            hidden_size: int,
            mlp_dim: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            *,
            rngs: nnx.Rngs = nnx.Rngs(0),
    ) -> None:

        self.norm1 = nnx.LayerNorm(hidden_size, rngs=rngs)
        #
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden_size,
            dropout_rate=dropout_rate,
            broadcast_dropout=False,
            decode=False,
            deterministic=False,
            rngs=rngs,
        )

        self.norm2 = nnx.LayerNorm(hidden_size, rngs=rngs)

        self.mlp = nnx.Sequential(
            nnx.Linear(hidden_size,
                       mlp_dim, rngs=rngs),

            nnx.gelu,
            nnx.Dropout(dropout_rate, rngs=rngs),

            nnx.Linear(mlp_dim, hidden_size, rngs=rngs),

            nnx.Dropout(dropout_rate, rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:

        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x
    
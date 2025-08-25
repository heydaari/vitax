from flax import nnx
import optax
import jax, jax.numpy as jnp


def compute_losses_and_logits(model: nnx.Module, images: jax.Array, labels: jax.Array):
    logits = model(images)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    ).mean()
    return loss, logits


@nnx.jit
def train_step(
        model: nnx.Module, optimizer: nnx.Optimizer, batch
):
    images = batch[0]
    labels = batch[1]

    grad_fn = nnx.value_and_grad(compute_losses_and_logits, has_aux=True)
    (loss, logits), grads = grad_fn(model, images, labels)

    optimizer.update(grads)  # In-place updates.

    return loss


@nnx.jit
def eval_step(
        model: nnx.Module, batch, eval_metrics: nnx.MultiMetric
):
    images = batch[0]
    labels = batch[1]
    loss, logits = compute_losses_and_logits(model, images, labels)

    eval_metrics.update(
        loss=loss,
        logits=logits,
        labels=labels,
    )



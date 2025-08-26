# vitax

[![PyPI version](https://badge.fury.io/py/vitax.svg)](https://badge.fury.io/py/vitax)

**vitax**: An open-source platform for training and inference of vanilla Vision Transformers (ViT) with the new and elegant **Flax NNX** API.

This library provides a clean, from-scratch implementation of the Vision Transformer model and makes it easy to leverage powerful pretrained models from the Hugging Face Hub for your own computer vision tasks.

## Core Features

*   **Modern Flax API**: Built entirely using `flax.nnx`, offering a more intuitive, object-oriented, and explicit way to build neural networks in JAX.
*   **Hugging Face Integration**: Seamlessly load pretrained ViT weights from `google/vit-*` models for transfer learning and fine-tuning.
*   **Custom Models**: Easily create and train Vision Transformer models from scratch with custom configurations.
*   **Simple & Efficient Training**: Includes a straightforward and JIT-compiled training and evaluation pipeline using `optax` for optimization.
*   **Modular Design**: The code is well-structured, separating the model definition, weight loading, and training logic for clarity and extensibility.

## Installation

You can install `vitax` directly from PyPI:

```bash
pip install vitax
```

You will also need to install the necessary peer dependencies:

```bash
pip install jax flax optax transformers datasets
```

## How to Use `vitax`

Using `vitax` is straightforward. You can create a model, load data, and start training in just a few steps.

### 1. Creating a Vision Transformer Model

The main entry point for creating a model is the `vitax.models.get_model` function.

#### Load a Pretrained Model for Fine-Tuning

This is the most common use case. You can load a model pretrained on ImageNet-21k and adapt its final classification layer for your specific dataset (e.g., CIFAR-100 with 100 classes).

```python
from vitax.models import get_model

# Load a base ViT model pretrained on ImageNet and adapt it for 100 classes
model = get_model(
    'google/vit-base-patch16-224',
    num_classes=100,
    pretrained=True
)

```

#### Create a Model from Scratch (Random Weights)

If you want to train a model from the ground up, you can create one with random weights. You can either use a standard configuration or define your own.

**Using a standard model configuration:**

```python
from vitax.models import get_model

# Create a 'vit-base-patch16-224' architecture with random weights
model = get_model(
    'google/vit-base-patch16-224',
    num_classes=10, # For a 10-class dataset like CIFAR-10
    pretrained=False
)

```

**Using a fully custom architecture:**

```python
from vitax.models import get_model

# Define a custom configuration for a smaller model
custom_config = {
    'image_size': 224,
    'patch_size': 16,
    'num_hidden_layers': 6,          # Fewer layers
    'num_attention_heads': 8,           # Fewer attention heads
    'intermediate_size': 2048,          # Smaller MLP dimension
    'hidden_size': 768,       # Embedding dimension
}

# Create the custom model with random weights
custom_model = get_model(
    name_or_config=custom_config,
    num_classes=10,
    pretrained=False
)

```

### 2. A Simple Training Pipeline (Fine-tuning on CIFAR-100)

Here is a complete example of how to fine-tune a pretrained `vitax` model on the CIFAR-100 dataset.

#### Step 1: Setup and Imports

First, let's import all the necessary libraries.

```python
import jax
import jax.numpy as jnp
import optax
import numpy as np
from flax import nnx
from datasets import load_dataset
from torchvision.transforms import v2 as T
import tqdm

# Import from the vitax library
from vitax.models import get_model
from vitax.training import train_step, eval_step
```

#### Step 2: Load and Preprocess Data

We'll use the `datasets` library to load CIFAR-100 and `torchvision.transforms` to apply standard data augmentations.

```python
# Load the dataset
train_dataset = load_dataset("cifar100", split="train")
val_dataset = load_dataset("cifar100", split="test")

# Define image transformations
IMG_SIZE = 224
MEAN = np.array([0.5, 0.5, 0.5])
STD = np.array([0.5, 0.5, 0.5])

def normalize(image):
    image = image.astype(np.float32) / 255.0
    return (image - MEAN) / STD

train_transforms = T.Compose([
    T.Lambda(lambda pil_image: np.asarray(pil_image.convert("RGB"))),
    T.Lambda(lambda np_array: jax.image.resize(np_array, (IMG_SIZE, IMG_SIZE), 'bicubic')),
    T.RandomHorizontalFlip(),
    T.Lambda(normalize),
])

val_transforms = T.Compose([
    T.Lambda(lambda pil_image: np.asarray(pil_image.convert("RGB"))),
    T.Lambda(lambda np_array: jax.image.resize(np_array, (IMG_SIZE, IMG_SIZE), 'bicubic')),
    T.Lambda(normalize),
])

def apply_transforms(batch):
    batch["img"] = [train_transforms(img) for img in batch["img"]]
    return batch

train_dataset.set_transform(apply_transforms)
val_dataset.set_transform(val_transforms)
```

#### Step 3: Create DataLoaders

We need a way to iterate over the data in batches. You can use any data loader you prefer. Here, we'll use a simple manual batching generator.

```python
def collate_fn(batch):
    images = np.stack([item['img'] for item in batch])
    labels = np.stack([item['fine_label'] for item in batch])
    return {'img': images, 'fine_label': labels}

def create_dataloader(dataset, batch_size, shuffle=True):
    indices = np.arange(len(dataset))
    if shuffle:
        np.random.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        if len(batch_indices) < batch_size and shuffle:
            continue # Drop last batch if shuffling
        yield collate_fn([dataset[int(j)] for j in batch_indices])

TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 128

train_loader = create_dataloader(train_dataset, TRAIN_BATCH_SIZE)
val_loader = create_dataloader(val_dataset, VAL_BATCH_SIZE, shuffle=False)
```

#### Step 4: Initialize Model and Optimizer

Now, we create our model for fine-tuning and set up the optimizer using `optax`.

```python
NUM_CLASSES = 100
LEARNING_RATE = 0.001
NUM_EPOCHS = 3

# Get a pretrained model adapted for CIFAR-100
model = get_model(
    'google/vit-base-patch16-224',
    num_classes=NUM_CLASSES,
    pretrained=True
)

# Setup the optimizer
optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=LEARNING_RATE))
```

#### Step 5: The Training Loop

We'll define helper functions for training one epoch and for evaluation. These will use the pre-built `train_step` and `eval_step` functions from `vitax`.

```python
# Create evaluation metrics
eval_metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average('loss'),
    accuracy=nnx.metrics.Accuracy(),
)

# For logging metrics
train_metrics_history = {"train_loss": []}
eval_metrics_history = {"val_loss": [], "val_accuracy": []}

total_steps = len(train_dataset) // TRAIN_BATCH_SIZE

def train_one_epoch(epoch):
    """Trains the model for one epoch."""
    model.train()  # Set model to training mode
    with tqdm.tqdm(
        desc=f"[Train] Epoch: {epoch+1}/{NUM_EPOCHS}",
        total=total_steps,
        leave=True,
    ) as pbar:
        # Create a new data loader for each epoch to handle shuffling
        train_loader = create_dataloader(train_dataset, TRAIN_BATCH_SIZE)
        for batch in train_loader:
            batch_tuple = (jnp.array(batch['img']), jnp.array(batch['fine_label']))
            loss = train_step(model, optimizer, batch_tuple)
            train_metrics_history["train_loss"].append(loss.item())
            pbar.set_postfix({"loss": loss.item()})
            pbar.update(1)

def evaluate_model(epoch):
    """Evaluates the model on the validation set."""
    model.eval()  # Set model to evaluation mode
    eval_metrics.reset()
    val_loader = create_dataloader(val_dataset, VAL_BATCH_SIZE, shuffle=False)
    for val_batch in val_loader:
        batch_tuple = (jnp.array(val_batch['img']), jnp.array(val_batch['fine_label']))
        eval_step(model, batch_tuple, eval_metrics)

    metrics = eval_metrics.compute()
    eval_metrics_history['val_loss'].append(metrics['loss'])
    eval_metrics_history['val_accuracy'].append(metrics['accuracy'])
    print(
        f"\n[Val] Epoch: {epoch+1}/{NUM_EPOCHS} | "
        f"Loss: {metrics['loss']:.4f} | "
        f"Accuracy: {metrics['accuracy']:.4f}"
    )

# --- Run The Training ---
print("Starting training...")
for epoch in range(NUM_EPOCHS):
    train_one_epoch(epoch)
    evaluate_model(epoch)

print("Training finished!")
```

## Contributing

Contributions are welcome! If you find a bug or have a feature request, please open an issue on the GitHub repository.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

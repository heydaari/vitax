# VITAX

[![PyPI version](https://badge.fury.io/py/vitax.svg)](https://badge.fury.io/py/vitax)

**VITAX**: An open-source platform for training and inference of vanilla Vision Transformers (ViT) with the new and elegant **Flax NNX** API.

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

# Load a base pretrained ViT model and adapt it for 100 classes
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

# Define a custom configuration for a smaller model, compatible with HuggingFace ViT config
custom_config = {
    'image_size': 224,
    'patch_size': 16,
    'num_hidden_layers': 6,          # Fewer layers
    'num_attention_heads': 8,           # Fewer attention heads
    'intermediate_size': 500,          # Smaller MLP dimension
    'hidden_size': 128,       # Embedding dimension
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
from datasets import load_dataset
import numpy as np
from torchvision.transforms import v2 as T

train_dataset = load_dataset("cifar100", split="train")
val_dataset = load_dataset("cifar100", split="test")

img_size = 224

def to_np_array(pil_image):
  return np.asarray(pil_image.convert("RGB"))

def normalize(image):
    # Image preprocessing matches the one of pretrained ViT
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    image = image.astype(np.float32) / 255.0
    return (image - mean) / std

tv_train_transforms = T.Compose([
    T.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0)), # Adjusted scale for smaller 32x32 images
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.Lambda(to_np_array),
    T.Lambda(normalize),
])

tv_test_transforms = T.Compose([
    T.Resize((img_size, img_size)),
    T.Lambda(to_np_array),
    T.Lambda(normalize),
])

def get_transform(fn):
    def wrapper(batch):
        
        batch["img"] = [
            fn(pil_image) for pil_image in batch["img"]
        ]
        return batch

    return wrapper

train_transforms = get_transform(tv_train_transforms)
val_transforms = get_transform(tv_test_transforms)

train_dataset = train_dataset.with_transform(train_transforms)
val_dataset = val_dataset.with_transform(val_transforms)
```

#### Step 3: Create DataLoaders

We need a way to iterate over the data in batches. You can use any data loader you prefer. Here, we'll use a grain dataloader objects.

```python
import grain.python as grain

seed = 12
train_batch_size = 32
val_batch_size = 2 * train_batch_size


train_sampler = grain.IndexSampler(
    len(train_dataset),  
    shuffle=True,            
    seed=seed,               
    shard_options=grain.NoSharding(),  
    num_epochs=1,            
)

val_sampler = grain.IndexSampler(
    len(val_dataset),  
    shuffle=False,         
    seed=seed,             
    shard_options=grain.NoSharding(),  
    num_epochs=1,          
)


train_loader = grain.DataLoader(
    data_source=train_dataset,
    sampler=train_sampler,                 
    worker_count=4,                        
    worker_buffer_size=2,                  
    operations=[
        grain.Batch(train_batch_size, drop_remainder=True),
    ]
)

val_loader = grain.DataLoader(
    data_source=val_dataset,
    sampler=val_sampler,                   
    worker_count=4,                        
    worker_buffer_size=2,
    operations=[
        grain.Batch(val_batch_size),
    ]
)
```

#### Step 4: Initialize Model and Optimizer

Now, we create our model for fine-tuning and set up the optimizer using `optax`.

```python
NUM_CLASSES = 100
LEARNING_RATE = 0.001
num_classes = 3

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

def train_one_epoch(epoch):

    model.train()  

    for batch in train_loader:
        batch = (jnp.array(batch['img']) , jnp.array(batch['fine_label']))

        loss = train_step(model, optimizer, batch)
        train_metrics_history["train_loss"].append(loss.item())


def evaluate_model(epoch):
    model.eval() 

    eval_metrics.reset() 
    for val_batch in val_loader:
        eval_step(model, (val_batch['img'], val_batch['fine_label']), eval_metrics)

    for metric, value in eval_metrics.compute().items():
        eval_metrics_history[f'val_{metric}'].append(value)

    print(f"[val] epoch: {epoch + 1}/{num_epochs}")
    print(f"- total loss: {eval_metrics_history['val_loss'][-1]:0.4f}")
    print(f"- Accuracy: {eval_metrics_history['val_accuracy'][-1]:0.4f}")

# And here goes the training
for epoch in range(num_epochs):
    train_one_epoch(epoch)
    evaluate_model(epoch)
```

## Contributing

Contributions are welcome! If you find a bug or have a feature request, please open an issue on the GitHub repository.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

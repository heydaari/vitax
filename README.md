# VITAX

[![PyPI version](https://badge.fury.io/py/vitax.svg)](https://badge.fury.io/py/vitax)

**VITAX**: An open-source platform for training and inference of Vision Transformers (ViT) with the new and elegant **Flax NNX** API.

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

## Model creation with Vitax

In Vitax, you can create vision transformers in different ways

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

## Contributing

Contributions are welcome! If you find a bug or have a feature request, please open an issue on the GitHub repository.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

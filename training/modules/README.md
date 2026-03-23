# Modules Documentation (training/modules)

This directory contains the refactored logic of the training pipeline, separated into specific modules to improve maintainability and code reuse.

## Table of Contents

| File | Description |
| :--- | :--- |
| `config.py` | **Configuration**: Global constants, file paths, and default parameters (Batch Size, LR, etc.). |
| `data_loaders.py` | **Data Loading**: Functions for creating DataLoaders (`get_dataloaders`) and splitting the dataset (`create_splits`). |
| `dataset.py` | **Dataset Class**: `ETL9Dataset` implementation for managing binarized ETL9B images. |
| `evaluation.py` | **Inference**: Functions for evaluation (`predict_and_evaluate`), error visualization, and Monte Carlo Dropout support. |
| `fonts.py` | **Fonts**: Utilities for loading Japanese fonts required for visualization (Matplotlib). |
| `image_processing.py` | **Processing**: Normalization/denormalization functions and image preprocessing (Otsu). |
| `models.py` | **Architectures**: PyTorch model definitions. Includes `MultiHeadKanjiClassificator` (MobileNetV3 with component head). |
| `optuna.py` | **Optimization**: `objective` function for hyperparameter search with Optuna. |
| `train_model.py` | **Training**: Main training (`train_model`) and validation loops. |
| `train_utils.py` | **Training Utilities**: Auxiliary classes such as `EarlyStopping` and optimizer/scheduler configuration. |
| `transforms.py` | **Augmentation**: Transformation and Data Augmentation pipeline definitions with `torchvision` and `albumentations`. |
| `utils.py` | **General**: General utility functions such as seed configuration (`set_seed`) and device detection. |
| `visualization.py` | **Visualization**: `TrainingPlotter` class for training plot generation (loss and accuracy). |

## General Usage

To use these modules from a notebook or script in the parent directory (`training/`), ensure the package is accessible (by default Python adds the current directory to the path):

```python
from modules.config import *
from modules.models import build_multi_head_model
# ...
```

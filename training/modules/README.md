# Modules Documentation

This directory contains the refactored implementation of the kanji model training pipeline. The notebook delegates most reusable logic to these modules in order to keep experimentation code concise and maintainable.

## Module Reference

| File | Description |
| :--- | :--- |
| `config.py` | Centralizes runtime flags, paths, artifact filenames, and default hyperparameters for local and Kaggle execution. |
| `data_loaders.py` | Creates deterministic dataset splits and PyTorch dataloaders, including a reduced-class loader pair for Optuna trials. |
| `dataset.py` | Indexes ETL9B image folders, filters valid kanji classes, and emits kanji, radical, and stroke labels for each sample. |
| `evaluation.py` | Runs folder-based inference, computes Top-1/Top-3/Top-5 accuracy, supports optional Monte Carlo Dropout, and visualizes predictions. |
| `fonts.py` | Loads a Japanese font into Matplotlib so kanji labels render correctly in plots. |
| `image_processing.py` | Applies Otsu binarization, inference-time preprocessing, and tensor denormalization for visualization. |
| `models.py` | Defines the `MultiHeadKanjiClassificator` based on a `timm` GhostNet backbone with radical and stroke auxiliary heads. |
| `optuna.py` | Defines the Optuna objective function and the reduced training loop used during hyperparameter search. |
| `train_model.py` | Implements the main epoch loop, curriculum gating for the kanji head, checkpointing, validation, and best-model restoration. |
| `train_utils.py` | Provides early stopping, optimizer and scheduler setup, loss construction, and curriculum management utilities. |
| `transforms.py` | Declares the data augmentation and validation preprocessing pipelines, including morphological operations and binarization. |
| `utils.py` | Provides reproducibility helpers and device selection across CUDA, MPS, and CPU backends. |
| `visualization.py` | Generates training dashboards for loss, kanji accuracy, radical accuracy, and stroke accuracy. |

## General Usage

These modules are intended to be imported from the parent `training/` directory:

```python
from modules.config import *
from modules.models import build_multi_head_model
from modules.train_model import train_model
```

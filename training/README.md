# Training Pipeline Documentation

This directory contains the main training workflow for the kanji recognition model. The notebook `kanji_classificator_model_training.ipynb` is the primary entry point and orchestrates dataset indexing, model construction, training, evaluation, and result visualization.

## Directory Overview

| File / Directory | Type | Description |
| :--- | :--- | :--- |
| `kanji_classificator_model_training.ipynb` | Notebook | Main end-to-end training and evaluation workflow. |
| `modules/` | Directory | Refactored Python modules used by the notebook. |
| `training_output/` | Directory | Saved weights, class mappings, and serialized training history. |
| `environment.yml` | Environment file | Conda environment definition used to reproduce the training setup. |
| `README.md` | Documentation | This document. |

## Notebook Entry Point

The notebook is organized as a staged pipeline:

1. Initial configuration and path loading.
2. ETL9B dataset indexing and split creation.
3. Model construction and summary.
4. Optional Optuna search.
5. Full training with checkpointing and best-model saving.
6. Plotting of loss and accuracy curves.
7. Test-set evaluation.
8. External inference and CASIA evaluation.

For implementation details, see `modules/` and the root `README.md`.

## Python Modules

| Module | Description |
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

## Running the Notebook

Create and activate a Conda environment from the exported definition, then start Jupyter from the repository root:

```bash
conda env create -f training/environment.yml
conda activate kanji-training
conda install jupyter
jupyter notebook training/kanji_classificator_model_training.ipynb
```

## Environment File Note

The environment definition used by this training pipeline is stored in `training/environment.yml`. If older documentation refers to an environment file at the repository root, that reference is outdated.

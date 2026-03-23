# Kanji Character Recognition Model Training Documentation

This document describes the workflow, architecture, and results of the training experiment recorded in the attached notebook. **Values shown here (hyperparameters, metrics, architecture  details) are manually maintained and may differ from the actual implementation. *The notebook output cells and source code are always the source of truth.***

## Overview
The notebook `kanji_classificator_model_training.ipynb` (formerly `train.ipynb`) implements a full Deep Learning cycle for handwritten character classification.

## Table of Contents

| File / Directory | Type | Description |
| :--- | :--- | :--- |
| `kanji_classificator_model_training.ipynb` | Notebook | Main training and evaluation pipeline. |
| `modules/` | Directory | Python modules used in the training pipeline. |
| `training_output/` | Directory | Training results (model, history, classes). |
| `training_output/last_checkpoint.pth` | File | Last training state for resumption. |
| `training_output/radical_classes.json` | File | Radical class mappings. |
| `training_output/stroke_classes.json` | File | Stroke count class mappings. |
| `README.md` | Doc | This documentation. |


## Notebook Stages

The notebook is structured into the following logical sections:

1.  **Configuration and Constants**:
    * Paths to data and output directories are established.
    * Global parameters governing the training (hyperparameters) are defined.
    * **Renamed**: The main notebook is now `kanji_classificator_model_training.ipynb`.
    * `MAX_CLASSES_LIMIT` is included to allow training with a reduced number of classes and a fixed seed is set for reproducibility.
    * Compute device (GPU/CPU) is configured.
    * **CASIA**: The CASIA dataset is used for external dataset testing on the kanji model.

2.  **Data Preparation and Preprocessing**:
    * Support for the **ETL9B** (binarized) dataset, replacing ETL9G.
    * Implementation of a dataset management class responsible for reading images and their labels, with support for filtering the maximum number of classes to use.
    * Definition of **transformations and Data Augmentation** see more in `modules/transform.py`:
    * **Balance Verification**: A cell using `Counter` to display each class frequency and verify if the dataset is balanced.

3.  **Architecture Definition**:
    * Loading a pre-trained base model and fine-tuning it with the ETL9B dataset.

4.  **Training Loop**:
    * Training execution for a defined number of epochs.
    * **Checkpoint System**: Automatic saving of training state (`last_checkpoint.pth`) at each epoch to allow resumption after interruptions.
    * Loss and accuracy calculation for both training and validation.
    * Automatic saving of the **best model** based on validation accuracy (`best_kanji_model.pth`).
    * Storage of metric history and class mappings.

5.  **Results Analysis**:
    * Visualization of learning curves (Loss and Accuracy) to diagnose model behavior (e.g., overfitting).

6.  **Evaluation and Inference**:
    * Final performance measurement using the Test subset (unseen during training).
    * Prediction tests with external images located in a specific folder.
    * **Preprocessing**: **Otsu** thresholding is incorporated to binarize inference inputs, ensuring compatibility with ETL9B-based training.
    * **Inference Improvements**: The `predict_and_evaluate` function now supports two image loading formats:
        * Filename as label (e.g., `あ.png`).
        * Containing folder as label (e.g., `あ/001.png`).
    * **Uncertainty**: The model can be extended to estimate uncertainty via techniques such as Monte Carlo Dropout if dropout layers are added during training.

7.  **Hyperparameter Optimization (Optuna)**:
    * **Optuna** has been integrated for automatic search of the best hyperparameters.
    * **Speed Optimization**: To accelerate the Optuna stage, only **2%** of the classes from the original dataset are used.
    * Definition of an objective function (`objective`) that trains the model with different configurations suggested by the "trial".
    * Optimization of variables such as: `learning_rate`, `batch_size`, `optimizer` (Adam/SGD), etc.
    * Visualization of hyperparameter importance and optimization history.

## Neural Network Architecture

* **Base Model**: GhostNet.
* **Heads (Multi-Head)**:
    * **Kanji Head**: Main Kanji character classification (primary classes).
    * **Radical Head**: Auxiliary classification of components/radicals.
    * **Stroke Head**: Auxiliary classification of stroke count.
* **Input**: Images resized to 128x128 pixels in 3 channels.
* **Strategy**: The backbone extracts features that are concatenated with radical and stroke predictions to feed the final Kanji head, improving accuracy by incorporating structural information.

## Hyperparameters

The following hyperparameters were used in this version of the experiment:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Learning Rate** | 0.0008189 | Learning rate (Optuna). |
| **Batch Size** | 128 | Batch size. |
| **Epochs** | 30 | Epochs executed. |
| **Image Size** | 128 x 128 | Input resolution (3 channels). |
| **Optimizer** | AdamW | AdamW optimizer (with Weight Decay). |
| **Architecture** | GhostNet | Efficiency-oriented architecture with Ghost modules. |

## General Results

In the execution recorded in this notebook using the **GhostNet** architecture, the following results were obtained:

* **Experiment ID**: `ghostnet-model-v4`
* **Validation Accuracy (Best)**: 99.54%
* **Validation Loss (Final)**: 0.0914
* **Test Accuracy (Final)**: 99.54%
* **Top-5 Test Accuracy**: 99.97%
* **CASIA Evaluation (Test Top-5)**: 82.91%
* **Notes**: The model uses a multi-head architecture (Kanji + Components) on a GhostNet backbone. The best hyperparameters found by Optuna (from the previous model) were used to maximize performance. High generalization capacity is observed on the ETL9B set and competitive results on CASIA.

## Modularization (Refactoring)

The main notebook code has been **recently refactored** to deepen separation of concerns into independent modules located in the `modules/` directory. Visualization utilities and training loop robustness have been improved.

The main modules are:
- `dataset.py`: ETL9 dataset management
- `train_model.py`: Training loop with checkpoints
- `evaluation.py`: Inference and evaluation with Monte Carlo Dropout
- `optuna.py`: Hyperparameter optimization
- `transforms.py`: Data Augmentation transformations.
- `models.py`: Neural network model architecture definitions.
- `visualization.py`: Training plot generation.

For more details, see the `modules/README.md` file.
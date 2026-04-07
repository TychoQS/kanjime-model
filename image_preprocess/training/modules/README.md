# Image Preprocessing Training Modules

This directory contains the individual Python modules that comprise the preprocessing segmentation model's training pipeline of the parent directory.

## Modules Breakdown

| File | Description |
| :--- | :--- |
| `architecture.py` | Defines the segmentation neural network. Implements a U-Net with a `tu-mobilenetv3_large_100` encoder using `segmentation_models_pytorch`. |
| `config.py` | Centralized configuration file containing hyperparameter values (learning rate, batch size, epochs), system-level variables (seeds, paths, devices), and dataset definitions. |
| `dataset.py` | Contains the `KanjiDataset` class inheriting from PyTorch's `Dataset`. Handles reading pairs of input RGB images and grayscale target masks, as well as applying albumentations transforms. |
| `evaluation.py` | Provides functions to evaluate model performance, primarily implementing an Intersection over Union (`calculate_iou`) metric for masks. |
| `transform.py` | Defines geometric and color data augmentation operations bridging `albumentations` with PyTorch dataset format (`ToTensorV2`). |
| `utils.py` | Contains generic helper utilities such as the `EarlyStopping` class to halt training and a `seed_everything` function to guarantee pipeline reproducibility across platforms. |
| `train_model.py` | Contains the main training loop and the test loop. |

## Usage
These modules are intended to be imported directly by the main training scripts located in the parent directory (`../`). If you need to make changes to model components (like editing data augmentations, hyper-parameters, or dataset folder structures), you should edit these specific files rather than the main training script.

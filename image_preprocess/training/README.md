# Binarization of Kanji in Complex Backgrounds via U-Net Semantic Segmentation
This directory contains the training pipeline for the u-net network training. The goal of this model is to perform background removal and character segmentation (binarization) to isolate the character before they are fed to the main Kanji classification model to improve the final classification accuracy when the kanji to classificate is from an uncontrolled environment (e.g. real world images).

## Overview
The training workflow utilizes a U-Net architecture to produce masks. The dataset used to train the network is the own crafted `synthetic_kanji_dataset`, consisting of RGB input images and their corresponding binary masks based on the kanji in complex backgrounds.

## Directory Structure

| Directory/File | Description |
| :--- | :--- |
| `modules/` | Directory containing core Python modules (architecture, config, dataset, evaluation, etc.). |
| `train.py` | Main script for running the model training |

## Neural Network Architecture
* **Base Architecture**: U-Net (via `segmentation_models_pytorch`)
* **Encoder**: `tu-mobilenetv3_large_100` (initialized with ImageNet weights)
* **Input**: 3-channel (RGB) images
* **Output**: 1-channel mask with a Sigmoid activation to produce values between 0.0 and 1.0.

## Hyperparameters
Refer to `modules/config.py` for exact up-to-date configurations. Baseline parameters:
* **Batch Size**: 64
* **Learning Rate**: 1e-4
* **Early Stopping Patience**: 10 epochs
* **Scheduler Patience**: 5 epochs
* **Weight Decay**: 1e-4

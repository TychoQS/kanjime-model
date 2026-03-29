# Image Preprocessing

Preprocessing pipeline for Kanji character images, aimed at improving input quality before inference with the trained model.

## Overview

This directory contains preprocessing scripts that apply a series of transformations to images. The pipeline includes grayscale conversion, adaptive contrast enhancement (CLAHE), bilateral filtering, Otsu thresholding, and morphological closing. It generates a comparative image showing all pipeline stages.

## Table of Contents

| File / Directory | Type | Description |
| :--- | :--- | :--- |
| `custom_binarization.py` | Script | Main binarization pipeline (Grayscale + CLAHE + Bilateral + Otsu). |
| `cut_preprocess.py` | Script | Applies CUT generator to transform scene text to clean handwriting domain. |
| `nst_preprocess.py` | Script | Applies Neural Style Transfer (NST) model to transfer Kanji images in natural scenes to the model's input domain. |
| `milyaev.py` | Script | Binarization proposed in https://ieeexplore.ieee.org/abstract/document/6628598. |
| `run_cut_training.sh` | Script | Automates CUT model training and post-training log plotting. |
| `models/` | Directory | Cloned repository of preprocessing models. |
| `samples/` | Directory | Collection of Kanji character images for testing. |
| `output/` | Directory | Organized visual processing results by script. |
| `utils/` | Directory | Utility scripts. |
| `README.md` | Doc | This documentation. |

## Custom Binarization

The `custom_binarization.py` script applies the following transformations sequentially:

1. **Original**: Input image loading.
2. **Grayscale**: Conversion to a single channel.
3. **CLAHE** *(conditional)*: Adaptive contrast enhancement, applied only if the image is detected as low contrast.
4. **Bilateral Filtering/NLMeans Denoising**: Noise reduction with improved edge preservation.
5. **Otsu/Sauvola/Niblack Thresholding**: Binarization.
6. **Morphological Operation**: Morphological closing to remove small noise within stroke interiors.

### Usage

To process an image and generate the stage comparison:

```bash
python custom_binarization.py <path_to_image>
```

**Example:**
```bash
python custom_binarization.py samples/1.jpeg
```

The script generates a comparative image in `output/custom_binarization/` that includes the stages: Grayscale, CLAHE (if needed), Bilateral Filter, Otsu, and Morphological Operation.

## Milyaev Binarization

The `milyaev.py` script implements a binarization algorithm based on graph-cut refinement:
- **Paper**: [Binarization of Color Document Images via Graph Cuts](https://ieeexplore.ieee.org/abstract/document/6628598)
- **Authors**: S. Milyaev, V. Lempitsky, Y. Boykov.
- **How it works**: Uses local estimation (Niblack) and the image Laplacian to build a graph whose energy optimization separates text from background.

### Usage

To process an image using Milyaev binarization:

```bash
python milyaev.py <path_to_image>
```

**Example:**
```bash
python milyaev.py samples/1.jpeg
```

## CUT 
The `cut_preprocess.py` script applies a CUT model to transform scene text to the model's input image domain.
- **Paper**: [Contrastive Learning for Unpaired  Image-to-Image Translation](https://arxiv.org/abs/2007.15651)
- **Authors**: Enrui Zhang, et al.
- **How it works**: Uses the Generator of a CUT model trained on the dataset to translate images from the scene text domain to the model's input image domain. The training is done with the official implementation https://github.com/taesungp/contrastive-unpaired-translation

### Usage

To process an image using the CUT model:

```bash
python cut_preprocess.py <path_to_image>
```

**Example:**
```bash
python cut_preprocess.py samples/1.jpeg
```
## NST
The `nst_preprocess.py` script applies a Neural Style Transfer model to translate Kanji images in natural scenes to the model's input domain.
- **Paper**: [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
- **Authors**: Justin Johnson, Alexandre Alahi, Li Fei-Fei.
- **How it works**: Uses a pre-trained Generator (TransformerNet) from the PyTorch examples to transform images from the scene text domain to the target writing domain.

### Usage

To process an image using the NST model:

```bash
python nst_preprocess.py <path_to_image>
```

**Example:**
```bash
python nst_preprocess.py samples/1.jpeg
```

### Note
Due to hardware incompatibilities (CUDA version didn't support the used libraries of the pytorch example and an exception was constantly being thrown), the model used for NST was trained using Kaggle. The training script execution command is trivial and can
be consulted the syntax in the commit version of the cited repository.

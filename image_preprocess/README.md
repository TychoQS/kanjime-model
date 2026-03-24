# Image Preprocessing

Preprocessing pipeline for Kanji character images, aimed at improving input quality before inference with the trained model.

## Overview

This directory contains preprocessing scripts that apply a series of transformations to images. The pipeline includes grayscale conversion, adaptive contrast enhancement (CLAHE), bilateral filtering, Otsu thresholding, and morphological closing. It generates a comparative image showing all pipeline stages.

## Table of Contents

| File / Directory | Type | Description |
| :--- | :--- | :--- |
| `custom_binarization.py` | Script | Main binarization pipeline (Grayscale + CLAHE + Bilateral + Otsu). |
| `milyaev.py` | Script | Binarization proposed in https://ieeexplore.ieee.org/abstract/document/6628598. |
| `preprocess_utils.py` | Module | Utility classes for saving results with stage mosaic support. |
| `models/` | Directory | Cloned repository of preprocessing models. |
| `samples/` | Directory | Collection of Kanji character images for testing. |
| `output/` | Directory | Organized visual processing results by script. |
| `README.md` | Doc | This documentation. |

## Pipeline Stages

The `custom_binarization.py` script applies the following transformations sequentially:

1. **Original**: Input image loading.
2. **Grayscale**: Conversion to a single channel.
3. **CLAHE** *(conditional)*: Adaptive contrast enhancement, applied only if the image is detected as low contrast.
4. **Bilateral Filtering**: Noise reduction with improved edge preservation.
5. **Otsu Thresholding**: Automatic binarization with optimal threshold.
6. **Morphological Operation**: Morphological closing to remove small noise within stroke interiors.

## Usage

To process an image and generate the stage comparison:

```bash
python custom_binarization.py <path_to_image>
```

**Example:**
```bash
python custom_binarization.py samples/1.jpeg
```

The script generates a comparative image in `output/custom_binarization/` that includes the stages: Grayscale, CLAHE (if needed), Bilateral Filter, Otsu, and Morphological Operation.

### Milyaev Binarization

The `milyaev.py` script implements a binarization algorithm based on graph-cut refinement:
- **Paper**: [Binarization of Color Document Images via Graph Cuts](https://ieeexplore.ieee.org/abstract/document/6628598)
- **Authors**: S. Milyaev, V. Lempitsky, Y. Boykov.
- **How it works**: Uses local estimation (Niblack) and the image Laplacian to build a graph whose energy optimization separates text from background.

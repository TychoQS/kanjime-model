# Image Preprocessing Models

## Overview

This directory contains the repositories of models used for image preprocessing. These subdirectories are ignored by git to keep the repository size manageable, but their metadata is documented here for replication.

## Table of Models

| Directory | Type | Description | Source Repository |
| :--- | :--- | :--- | :--- |
| `CUT/` | Directory | Contrastive Unpaired Translation (transferred handwriting domain). | [taesungp/contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation) |
| `example_pytorch/` | Directory | Official PyTorch examples (includes Fast Style Transfer). | [pytorch/examples](https://github.com/pytorch/examples) |
| `README.md` | Doc | This documentation. | - |

## Replication

To replicate the current state of this directory, you can use the following commands from within this folder (`image_preprocess/models`):

```bash
# 1. Clone Contrastive Unpaired Translation (CUT)
git clone https://github.com/taesungp/contrastive-unpaired-translation CUT
cd CUT && git checkout b3ac297708dfb6f7589d04662277e53c0d579c27 && cd ..

# 2. Clone PyTorch Examples (for NST)
git clone https://github.com/pytorch/examples.git example_pytorch
cd example_pytorch && git checkout acc295dc7b90714f1bf47f06004fc19a7fe235c4 && cd ..
```

> [!IMPORTANT]
> Some models may require additional setup or downloading pre-trained weights. Refer to the specific README files within each cloned directory for detailed instructions.

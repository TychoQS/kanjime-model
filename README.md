# Kanji Recognition - Japanese Character Recognition

Deep Learning system for handwritten Kanji character recognition.

## Overview

This project implements a complete Machine Learning pipeline for classifying Kanji characters from the ETL9B dataset. It includes data preprocessing, training with data augmentation, hyperparameter optimization with Optuna, and evaluation with external images.

## Repository Structure

| Directory | Description |
| :--- | :--- |
| `training/` | Model training notebooks, modules, and outputs (`.pth`). |
| `inference_samples/` | External handwritten images for inference testing. |
| `image_preprocess/` | Image preprocessing pipeline. |
| `data/` | Datasets (not included in the repository due to size). |
| `resources/` | External resources (fonts, etc.). |
| `stats/` | Historical statistics and experiment records. |
| `.antigravity/` | AI agent configuration and rules used for documentation. |

# Known Issues and Fixes

## Test Accuracy Discrepancy (Commit 6216312) [SOLVED]
Test results reported in commits prior to `6216312` may differ from the values shown in the notebooks due to an evaluation error.

**Problem identified:**
- The `train_model()` and `train_kaggle()` functions returned the model from the **last epoch** instead of the **best model** saved to disk.
- Test evaluations were performed on the wrong model, causing accuracy variations of ±2-4%.

**Solution applied:**
- Modified `train_model()` and `train_kaggle()` to automatically load and return the best model.
- All subsequent evaluations use the model with the best validation accuracy.

**Impact:**
- Pre-fix results: Variable test accuracy (e.g., 90-94%).
- Post-fix results: Stable and reproducible test accuracy (e.g., 92.4%).

**Note:** This issue only affects test results. Evaluation results with other datasets were not affected since they loaded the model before evaluating.

## Validation Curve Instability and Overfitting (Commit 3444174) [SOLVED]
As more complex Data Augmentation (DA) transformations were added, validation curves began showing erratic behavior and a growing gap relative to the training curve (overfitting).

**Problem identified:**
- In early simple models, training was stable, but after introducing `Erode`, `Dilate`, and `Elastic` aggressively, validation became unstable.
- Certain transformations were altering character distributions in ways that caused the model to learn irrelevant artifacts for the ETL9B dataset (which is binarized).

**Solution applied:**
- Explicitly identified and removed transformations that generated the most noise in the binary domain and shifted the validation distribution away from training:
    - `ColorJitter` (Incompatible with the binarized nature of the dataset).
    - `GaussianNoise` (Generated pixel distributions inconsistent with the clean background).
- Adjusted `ElasticTransform` parameters to be less aggressive.

**Impact:**
- **Stabilization:** Validation curves went from erratic to closely tracking training trends.
- **Improved Generalization:** Models trained after this adjustment show much more consistent metrics on both the proprietary test set and external evaluations (CASIA).

## Missing v5 Versions in Commit History (Commit c56c90e) [SOLVED]
The tags `ghotnet-model-v5` and `ghostnet-model-v5-log` do not appear in the commit history or log file because these versions were incorrectly uploaded to the repository and their commits were deleted.

**Problem identified:**
- The versions tagged as `ghotnet-model-v5` and `ghostnet-model-v5-log` were uploaded to the repository with errors, so their commits were removed from history.
- As a consequence, these entries do not appear in the log file and the history shows a direct jump from version 4 to version 6 at commit `c56c90e`.

**Solution applied:**
- The problematic commits associated with v5 versions were deleted to maintain repository integrity.
- Development continued from version 4, with version 6 published directly containing the accumulated and corrected changes.
- The tags `ghotnet-model-v5` and `ghostnet-model-v5-log` have been preserved in the repository to allow access to those version files if needed.

**Impact:**
- The commit history and log file show a jump from v4 to v6, which is expected and does not indicate any loss of functionality.
- The results and repository state from commit `c56c90e` onwards are correct and consistent.

**Note:** The absence of v5 entries in the logs is intentional. Any reference to `ghotnet-model-v5` or `ghostnet-model-v5-log` should be considered obsolete for tracking purposes, although the tags remain accessible in the repository.

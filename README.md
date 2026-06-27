# Kanji Recognition Model Training Repository

Research and experimentation repository for the training pipeline of the Japanese kanji optical recognition model developed as part of a Computer Engineering Bachelor's Thesis at ULPGC. The trained model is intended to be exported to ONNX for integration into the KanjiMe mobile application.

## Repository Structure

| Directory | Description |
| :--- | :--- |
| `training/` | Main training pipeline, including the primary notebook and Python modules. |
| `training/modules/` | Refactored Python modules that implement the training workflow. |
| `training/training_output/` | Generated artifacts such as trained weights, class mappings, and metric history. |
| `data/` | Datasets used by the project. Large datasets are not expected to be versioned as part of the repository. |
| `image_preprocess/` | Image preprocessing experiments for domain adaptation and binarization. |
| `text_detection/` | Text detection experiments and detector benchmarks. |
| `inference_samples/` | Sample handwritten images used for inference checks. |
| `resources/` | Auxiliary resources, including typography assets used for visualization. |
| `stats/` | Training logs and benchmark summaries. |

## Neural Network Architecture

The current notebook and Python modules implement a multi-head convolutional architecture specialized for large-scale handwritten kanji classification.

* **Base model**: `GhostNet` from `timm`, instantiated as `ghostnet_100` with pretrained backbone weights and `num_classes=0`.
* **Input resolution**: `128 x 128` pixels.
* **Input channels**: `3` channels.
* **Auxiliary heads**:
  * **Radical head**: linear classifier over the backbone feature vector.
  * **Stroke head**: linear classifier over the same backbone feature vector.
* **Main kanji head**: linear classifier that receives the concatenation of:
  * backbone feature vector,
  * radical logits,
  * stroke logits.
* **Concatenation strategy**: the auxiliary predictions are not used only as side outputs; they are concatenated with the shared backbone features and fed into the final kanji classifier to inject structural information into the main decision layer.

## General Results

The current notebook configuration, `training/training_output/training_history.json`, and the corresponding entry in `stats/training_log.csv` align with the `ghostnet-model-v7` experiment.

| Field | Value |
| :--- | :--- |
| Experiment ID | `ghostnet-model-v7` |
| Validation Accuracy (best) | `99.61%` |
| Validation Loss (final) | `1.2650` |
| Test Accuracy | `99.56%` |
| Top-5 Test Accuracy | `99.96%` |
| CASIA Evaluation (Top-5) | `86.16%` |
| Notes | Multi-head GhostNet model trained on the full ETL9B class set (`2965` classes). The run uses the hyperparameters currently stored in `training/modules/config.py` and recorded as the latest full-scale training entry in `stats/training_log.csv`. |

## Hyperparameters

The following values are the active training configuration defined by the current notebook and `training/modules/config.py`.

| Parameter | Value | Source / Notes |
| :--- | :--- | :--- |
| Random seed | `42` | Reproducibility seed. |
| Maximum classes limit | `None` | Full ETL9B class set. |
| Image size | `128 x 128` | Input resolution used by the notebook. |
| Input channels | `3` | RGB conversion before tensorization. |
| Batch size | `96` | Default training batch size. |
| Epochs | `30` | Main training schedule. |
| Learning rate | `0.002724405295516834` | Default notebook value when Optuna is disabled. |
| Weight decay | `0.00016836131495616855` | `AdamW` regularization term. |
| Optimizer | `AdamW` | Configured in `setup_training_tools()`. |
| Scheduler | `ReduceLROnPlateau` | `factor=0.5`, `patience=2`, `min_lr=1e-6`. |
| Early stopping patience | `5` | Validation-accuracy based stopping criterion. |
| Kanji loss | `CrossEntropyLoss(label_smoothing=0.1)` | Main task loss. |
| Radical loss | `CrossEntropyLoss()` | Auxiliary task loss. |
| Stroke loss | `CrossEntropyLoss()` | Auxiliary task loss. |
| Radical loss weight | `0.4556013847773026` | `LAMBDA_RAD`. |
| Stroke loss weight | `0.5297980180240289` | `LAMBDA_STR`. |
| Train / validation / test split | `80% / 10% / 10%` | Produced by `create_splits()`. |
| Optuna trials | `10` | Used only when `OPTUNA_ENABLED=True`. |
| Optuna epochs per trial | `20` | Reduced schedule for search. |
| Training augmentations | `RandomPerspective`, `RandomAffine`, `ElasticTransform`, `GaussianBlur`, `MorphologicalTransform`, binarization, `RandomErasing` | Implemented in `training/modules/transforms.py`. |

## Environment Setup

The execution environment is managed with Conda. Create the environment from the exported file and activate it with a descriptive generic name:

```bash
conda env create -f training/environment.yml
conda activate kanji-training
```

To execute the main notebook, install Jupyter in the active environment and launch it directly from the repository root:

```bash
conda install jupyter
jupyter notebook training/kanji_classificator_model_training.ipynb
```

## Datasets

Datasets are not guaranteed to be included in a portable form in this repository and should be obtained manually from their official or documented sources before reproducing the experiments.

| Dataset | Role in the project | Source |
| :--- | :--- | :--- |
| `ETL9B` | Main training dataset for the current kanji classifier. | AIST ETL Character Database. |
| `ETL9G` | Earlier dataset version retained for previous experiments and comparisons. | AIST ETL Character Database. |
| `CASIA HWDB` | External evaluation dataset used to measure cross-dataset generalization. | CASIA Handwritten Chinese Character Database. |
| `KanjiVG` | Structural reference used to derive radicals and stroke counts. | KanjiVG project. |
| `JPSC1400 - Japanese Scene Character Dataset` | Additional handwriting dataset used elsewhere in the repository for preprocessing and domain-transfer experiments. | IMGLab Character Databases |
| `synthetic_kanji_dataset` | Synthetic kanji samples for auxiliary experimentation. | Generated locally by repository utilities. |
| `unpaired_jpsc1400_etl9b` | Unpaired dataset built for image-to-image translation experiments. | Generated locally from JPSC1400 and ETL9B assets. |
| `coco-training-2014` | Background imagery for synthetic or preprocessing workflows. | COCO 2014 dataset. |
| `dtd` | Texture dataset used for background generation workflows. | Describable Textures Dataset (DTD). |
| `icdar-2019` | Text-related dataset used in text detection experiments. | ICDAR 2019 dataset. |

## Notebook Stages

The main notebook, `training/kanji_classificator_model_training.ipynb`, is the entry point of the training pipeline and is organized into the following stages:

1. **Initial configuration**
   Loads the execution environment, seeds, paths, logging, and device selection.
   Related modules: `training/modules/config.py`, `training/modules/utils.py`.
2. **Dataset and model setup**
   Builds ETL9B datasets, applies transforms, creates train/validation/test splits, and instantiates the multi-head GhostNet model.
   Related modules: `training/modules/dataset.py`, `training/modules/transforms.py`, `training/modules/data_loaders.py`, `training/modules/models.py`.
3. **Dataset distribution check**
   Verifies whether all indexed classes contain the same number of samples.
   Related modules: `training/modules/dataset.py`.
4. **Optuna hyperparameter optimization**
   Optionally runs reduced-data hyperparameter search with a separate objective function and reduced loaders.
   Related modules: `training/modules/optuna.py`, `training/modules/data_loaders.py`, `training/modules/train_model.py`, `training/modules/train_utils.py`.
5. **Model training**
   Builds dataloaders, optimizer, scheduler, losses, and early stopping, then executes the full training loop with checkpointing and best-model persistence.
   Related modules: `training/modules/train_model.py`, `training/modules/train_utils.py`.
6. **Training metrics**
   Loads the serialized history and renders loss and accuracy plots for the three tasks.
   Related modules: `training/modules/visualization.py`.
7. **Test evaluation**
   Computes test-set accuracy, Top-3 accuracy, Top-5 accuracy, and error summaries on the held-out split.
   Related modules: `training/modules/evaluation.py`, `training/modules/image_processing.py`, `training/modules/fonts.py`.
8. **Input distribution inspection**
   Compares pixel histograms from training, validation/test, and inference preprocessing outputs.
   Related modules: `training/modules/image_processing.py`.
9. **External inference**
   Reloads the saved model and evaluates handwritten samples from `inference_samples/`.
   Related modules: `training/modules/evaluation.py`, `training/modules/models.py`.
10. **CASIA evaluation**
    Reuses the same inference pipeline on CASIA samples that overlap with the trained class vocabulary.
    Related modules: `training/modules/evaluation.py`.

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

## Main Dependencies

The environment definition is stored in `training/environment.yml`. The table below lists the most relevant project dependencies only.

| Dependency | Version |
| :--- | :--- |
| Python | `3.10.19` |
| PyTorch | `2.9.1` |
| torchvision | `0.24.1` |
| timm | `1.0.24` |
| ONNX | `1.20.1` |
| onnxruntime | `1.23.2` |
| OpenCV | `4.5.4.60` (`opencv-python`) |
| Optuna | `4.7.0` |
| NumPy | `1.26.4` |
| Pillow | `12.0.0` |
| scikit-image | `0.25.2` |
| matplotlib | `3.10.8` |
| seaborn | `0.13.2` |
| pandas | `2.3.3` |
| scikit-learn | `1.7.2` |
| tqdm | `4.67.1` |
| ipykernel | `7.1.0` |

## Known Issues and Fixes

### Test Accuracy Discrepancy (Commit 6216312) [SOLVED]

Test results reported in commits prior to `6216312` may differ from the values shown in the notebooks due to an evaluation error.

**Problem identified:**

* The `train_model()` and `train_kaggle()` functions returned the model from the last epoch instead of the best model saved to disk.
* Test evaluations were performed on the wrong model, causing accuracy variations of approximately `±2-4%`.

**Solution applied:**

* `train_model()` and `train_kaggle()` were modified to automatically load and return the best saved model.
* Subsequent evaluations therefore use the model with the highest validation accuracy.

**Impact:**

* Pre-fix results showed variable test accuracy.
* Post-fix results became stable and reproducible.

**Note:** This issue only affected test results. External evaluations were not affected because they already reloaded the stored model before evaluation.

### Validation Curve Instability and Overfitting (Commit 3444174) [SOLVED]

As more aggressive augmentation was introduced, validation curves became unstable and increasingly separated from the training curve.

**Problem identified:**

* Some transformations were too disruptive for binarized ETL9B samples.
* `ColorJitter` and strong additive noise were particularly harmful in this binary-domain setting.

**Solution applied:**

* `ColorJitter` was removed from the effective pipeline.
* Gaussian noise was disabled in the current transform definition.
* Elastic deformation was retained with controlled parameters.

**Impact:**

* Validation trends became much more stable.
* Generalization on ETL9B and CASIA improved in later GhostNet experiments.

### Missing v5 Versions in Commit History (Commit c56c90e) [SOLVED]

The tags `ghotnet-model-v5` and `ghostnet-model-v5-log` do not appear in the linear commit history because the corresponding uploads were faulty and their commits were removed.

**Problem identified:**

* The v5 artifacts were uploaded incorrectly.
* Their commits were deleted from history after the issue was detected.

**Solution applied:**

* Development continued from version 4 directly to version 6 with corrected artifacts.
* The tags were preserved only as historical references.

**Impact:**

* The apparent jump from v4 to v6 in the log is expected.
* The repository state from commit `c56c90e` onward is the valid reference.

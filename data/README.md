# Data Directory Reconstruction Guide

The `data/` directory is not versioned as part of this repository because the datasets are too large to commit. It must be reconstructed manually before running the training, preprocessing, or benchmarking experiments documented in this project.

## Expected Directory Structure

The following table describes the expected first-level subdirectories currently used in this workspace.

| Subdirectory | Dataset / Content | Official source | Expected format |
| :--- | :--- | :--- | :--- |
| `etlcb/` | ETL Character Database assets, including `ETL9B` for the main training pipeline and `ETL9G` for earlier experiments. | AIST ETL Character Database | The training code uses `data/etlcb/ETL9B`. Inside `ETL9B` and `ETL9G`, the effective dataset must be unpacked into folders ending in `_unpack`, each containing a `.csv` label file and numbered `.png` images such as `00001.png`. |
| `chinese-handwriting/` | CASIA Handwritten Chinese Character Database assets used for external evaluation. | CASIA Handwritten Chinese Character Database | Expected to contain raw `.gnt` files under `CASIA_HWDB_DATASET/` and extracted sample images under `CASIA_HWDB_SAMPLES/`, which is the path used by the evaluation pipeline. |
| `kanjiVG/` | KanjiVG structural reference data used to derive radicals and stroke counts. | KanjiVG project | Expected to contain `kanji_standard.json` and the KanjiVG SVG corpus under `kanji/`. |
| `JPSC1400-20201218/` | JPSC1400 Japanese scene character dataset used in preprocessing and domain-transfer experiments. | IMGLab Character Databases | Observed layout includes metadata files such as `README.txt` and `label.txt`, plus image directories such as `png/` and `ppm/`. |
| `coco-training-2014/` | COCO training images used as background imagery in synthetic or preprocessing workflows. | COCO 2014 dataset | Expected extracted image tree with `train2014/` containing `.jpg` files. |
| `dtd/` | Describable Textures Dataset used for texture and background generation workflows. | Describable Textures Dataset (DTD) | Expected extracted DTD structure including `images/` and related metadata directories; this workspace also contains derived assets such as `backgrounds_standalone/`. |
| `icdar-2019/` | ICDAR 2019 text-related dataset used in text detection experiments. | ICDAR 2019 dataset | Expected extracted dataset directories such as `icdar-2019-japanese/` with image files and paired text annotation files. |
| `synthetic_kanji_dataset/` | Locally generated synthetic kanji dataset used for auxiliary image preprocessing experiments. | Generated locally by repository utilities | Expected split-style layout such as `train/`, `val/`, and `test/`, alongside generator scripts or auxiliary metadata. |
| `unpaired_jpsc1400_etl9b/` | Locally generated unpaired dataset derived from JPSC1400 and ETL9B for image-to-image translation experiments. | Generated locally from JPSC1400 and ETL9B assets | Expected CUT-style domain folders such as `trainA/`, `trainB/`, `testA/`, and `testB/`, plus local generation scripts or analysis artifacts. |

## Notes on Format Expectations

- The main training pipeline reads from `data/etlcb/ETL9B`, as defined in `training/modules/config.py`.
- The `ETL9Dataset` class scans only subdirectories whose names end with `_unpack`, then reads a `.csv` file and numbered `.png` files from each unpacked shard.
- External handwritten evaluation uses `data/chinese-handwriting/CASIA_HWDB_SAMPLES`, while the raw CASIA `.gnt` files may be retained separately in the same top-level directory.
- Structural annotations for radicals and stroke counts are expected at `data/kanjiVG/kanji_standard.json`.

## Example Layout

Once all datasets have been downloaded, unpacked, and locally generated where required, the directory structure should resemble the following:

```bash
$ tree -L 2 data/
data/
├── JPSC1400-20201218
│   ├── png
│   └── ppm
├── chinese-handwriting
│   ├── CASIA_HWDB_DATASET
│   └── CASIA_HWDB_SAMPLES
├── coco-training-2014
│   └── train2014
├── dtd
│   ├── backgrounds_standalone
│   └── images
├── etlcb
│   ├── ETL9B
│   └── ETL9G
├── icdar-2019
│   └── icdar-2019-japanese
├── kanjiVG
│   └── kanji
├── synthetic_kanji_dataset
│   ├── test
│   ├── train
│   └── val
└── unpaired_jpsc1400_etl9b
    ├── testA
    ├── testB
    ├── trainA
    └── trainB
```

`JPSC1400-20201218/` contains the JPSC1400 dataset assets used by `image_preprocess/benchmark.py`. That script reads `label.txt` as the annotation file and loads the source images from `png/` using identifiers from the label file, expecting filenames such as `0001.png`, `0002.png`, and similar zero-padded image names. The sibling `ppm/` directory is part of the observed repository layout, but the preprocessing benchmark consumes the `png/` branch specifically.

`chinese-handwriting/` contains the CASIA evaluation assets used by the external evaluation path configured in `training/modules/config.py` and consumed by `training/modules/evaluation.py`. The raw dataset is stored under `CASIA_HWDB_DATASET/` as `.gnt` files with names such as `001-f.gnt` and `006-t.gnt`. The evaluation code reads from `CASIA_HWDB_SAMPLES/`, where the repository currently stores class-specific subfolders and numbered `.png` files such as `CASIA_HWDB_SAMPLES/一/0.png`. These extracted sample images are generated from the raw `.gnt` files by `data/chinese-handwriting/casia_hwdb_samples_builder.py`, which also writes `stats_values.txt`.

`coco-training-2014/` contains COCO background images used indirectly by the synthetic dataset workflow documented for image preprocessing. The observed second-level directory is `train2014/`, which stores `.jpg` files named according to the COCO convention, for example `COCO_train2014_000000000009.jpg`. Although the training-side kanji modules do not consume this subtree directly, it is part of the background-image resources expected by the synthetic data tooling described elsewhere in the repository.

`dtd/` contains the Describable Textures Dataset resources used by the synthetic preprocessing dataset generator in `data/synthetic_kanji_dataset/synthetic_dataset_generator.py`. That script reads backgrounds from `../dtd/backgrounds_standalone`, while the repository also keeps the extracted DTD image corpus under `images/`. The helper script `data/dtd/generate_background_standalone.sh` generates `backgrounds_standalone/` by copying image files such as `.jpg`, `.jpeg`, and `.png` from `images/`, producing filenames such as `banded_0002.jpg`.

`etlcb/` contains the ETL character database assets used by the main training pipeline. `training/modules/config.py` points `DATASET_PATH` to `data/etlcb/ETL9B`, and `training/modules/dataset.py` expects that dataset root to contain subdirectories ending in `_unpack`. Inside each unpacked shard, the loader looks for a `.csv` annotation file and numbered `.png` files built with the `f"{int(row.iloc[0]):05d}.png"` convention, such as `00001.png`. The same layout is expected under `ETL9G/` for earlier experiments, while the repository also contains raw source files such as `ETL9B_1`, `ETL9G_01`, and `ETL9INFO` before unpacking.

`icdar-2019/` contains ICDAR 2019 text-detection assets used by the text-detection experiment area. The observed second-level directory is `icdar-2019-japanese/`, which stores image files such as `tr_img_04003.jpg` together with paired text annotation files such as `tr_img_04003.txt`, and also includes `classes.txt`. This subtree is part of the text-oriented experimental data inventory referenced by the repository documentation, even though it is not consumed by `training/modules/config.py` or `training/modules/dataset.py`.

`kanjiVG/` contains the structural reference data used by the training pipeline to derive radicals and stroke counts. `training/modules/config.py` points `KANJI_STRUCTURE_JSON_PATH` to `data/kanjiVG/kanji_standard.json`, and `training/modules/dataset.py` loads that JSON file to map each kanji to its radical and stroke metadata. The second-level `kanji/` directory stores the underlying SVG corpus with filenames such as `00021.svg`, `0002c.svg`, and related hexadecimal codepoint-based names. The repository also includes local helper scripts such as `kanjivg_utils.py` and `counter.py` alongside this data.

`synthetic_kanji_dataset/` contains the locally generated synthetic dataset used by the U-Net preprocessing training workflow configured in `image_preprocess/training/modules/config.py`, where `DATASET_ROOT` is set to `../../data/synthetic_kanji_dataset`. The generator script `data/synthetic_kanji_dataset/synthetic_dataset_generator.py` creates the split directories `train/`, `val/`, and `test/`; within each split it creates `images/` and `masks/` subdirectories, writing `.jpg` scene images named with integer identifiers such as `900000.jpg` and mask files named with the `{id}_mask.png` convention. This subtree is therefore generated locally by repository tooling rather than downloaded as-is from an external source.

`unpaired_jpsc1400_etl9b/` contains the locally generated unpaired dataset used by the CUT training workflow. `image_preprocess/run_cut_training.sh` passes `../data/unpaired_jpsc1400_etl9b` as the CUT `--dataroot`, and the generated structure follows the expected CUT layout with `trainA/`, `trainB/`, `testA/`, and `testB/`. The generator script `data/unpaired_jpsc1400_etl9b/generate_unpaired_dataset.py` builds these folders from JPSC1400 and ETL9B samples and copies images using six-digit zero-padded `.png` filenames such as `000000.png`. The same subtree also stores local analysis artifacts such as `dataset_analysis_report.json`, produced by the generation workflow.

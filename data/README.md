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

Once all datasets have been downloaded, unpacked, and locally generated where required, the first-level structure should resemble the following:

```bash
$ tree -L 1 data/
data/
├── JPSC1400-20201218
├── chinese-handwriting
├── coco-training-2014
├── dtd
├── etlcb
├── icdar-2019
├── kanjiVG
├── synthetic_kanji_dataset
└── unpaired_jpsc1400_etl9b
```

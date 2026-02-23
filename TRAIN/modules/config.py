import os
import sys

# Seed
RANDOM_SEED = 42

# Execution flags
QUICK_TEST = False
OPTUNA_ENABLED = False
ON_KAGGLE = os.path.exists('/kaggle/input')

# Paths
if ON_KAGGLE:
    # Kaggle paths
    modules_path = '/kaggle/input/modules'
    if modules_path not in sys.path:
        sys.path.append(modules_path)
    
    DATASET_PATH = '/kaggle/input/etl9b-dataset-full/ETL9B'
    PREDICTION_FOLDER = '/kaggle/input/tests-images/TESTS'
    CASIA_HWDB_DATASET_SAMPLES_PATH = "/kaggle/input/chinese-handwriting/CASIA_HWDB_SAMPLES"
    FONT_PATH = '/kaggle/input/noto-sans-jp/Noto_Sans_JP/static/NotoSansJP-Regular.ttf'
    OUTPUT_DIR = '/kaggle/working/Training_Output'
    KANJI_STRUCTURE_JSON_PATH = '/kaggle/working/kanjiVG/kanji_standard.json'
else:
    # Local paths
    DATASET_PATH = '../DATA/etlcb/ETL9B'
    PREDICTION_FOLDER = '../TESTS'
    CASIA_HWDB_DATASET_SAMPLES_PATH = "../DATA/chinese-handwriting/CASIA_HWDB_SAMPLES"
    FONT_PATH = '../Noto_Sans_JP/static/NotoSansJP-Regular.ttf'
    OUTPUT_DIR = './Training_Output'
    KANJI_STRUCTURE_JSON_PATH = '../DATA/kanjiVG/kanji_standard.json'

# Filenames
MODEL_FILENAME = 'best_kanji_model.pth'
HISTORY_FILENAME = 'training_history.json'
CLASSES_FILENAME = 'classes.json'
RADICAL_CLASSES_FILENAME = 'radical_classes.json'
STROKE_CLASSES_FILENAME = 'stroke_classes.json'

# Full file paths
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, MODEL_FILENAME)
HISTORY_SAVE_PATH = os.path.join(OUTPUT_DIR, HISTORY_FILENAME)
CLASSES_SAVE_PATH = os.path.join(OUTPUT_DIR, CLASSES_FILENAME)
RADICAL_SAVE_PATH = os.path.join(OUTPUT_DIR, RADICAL_CLASSES_FILENAME)
STROKE_SAVE_PATH = os.path.join(OUTPUT_DIR, STROKE_CLASSES_FILENAME)
MODEL_PATH = MODEL_SAVE_PATH

# Hyperparameters
if QUICK_TEST:
    MAX_CLASSES_LIMIT = 5
    BATCH_SIZE = 2
    NUM_EPOCHS = 1
    OPTUNA_TRIALS = 4
    OPTUNA_EPOCHS = 1
else:
    MAX_CLASSES_LIMIT = None
    BATCH_SIZE = 96         
    NUM_EPOCHS = 30
    OPTUNA_TRIALS = 10
    OPTUNA_EPOCHS = 20
CHANEL_SIZE = 3 
LEARNING_RATE = 0.002724405295516834
WEIGHT_DECAY = 0.00016836131495616855
IMG_SIZE = 128
LAMBDA_RAD = 0.4556013847773026
LAMBDA_STR = 0.5297980180240289

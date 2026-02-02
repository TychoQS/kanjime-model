import os
import sys

# Seed
RANDOM_SEED = 42

# Execution flags
QUICK_TEST = False
OPTUNA_ENABLED = True
ON_KAGGLE = os.path.exists('/kaggle/input')

# Paths
if ON_KAGGLE:
    # Kaggle paths
    modules_path = '/kaggle/input/modules'
    if modules_path not in sys.path:
        sys.path.append(modules_path)
    
    DATASET_PATH = '/kaggle/input/etl9b-dataset-full/ETL9B'
    PREDICTION_FOLDER = '/kaggle/input/tests-images/TESTS'
    CASIA_DATASET_TRAIN_PATH = "/kaggle/input/chinese-handwriting/CASIA-HWDB_Train/Train"
    CASIA_DATASET_TEST_PATH = "/kaggle/input/chinese-handwriting/CASIA-HWDB_Test/Test"
    FONT_PATH = '/kaggle/input/noto-sans-jp/Noto_Sans_JP/static/NotoSansJP-Regular.ttf'
    OUTPUT_DIR = '/kaggle/working/Training_Output'
else:
    # Local paths
    DATASET_PATH = '../DATA/etlcb/ETL9B'
    PREDICTION_FOLDER = '../TESTS'
    CASIA_DATASET_TRAIN_PATH = "../DATA/chinese-handwriting/CASIA-HWDB_Train/Train"
    CASIA_DATASET_TEST_PATH = "../DATA/chinese-handwriting/CASIA-HWDB_Test/Test"
    FONT_PATH = '../Noto_Sans_JP/static/NotoSansJP-Regular.ttf'
    OUTPUT_DIR = './Training_Output'

# Filenames
MODEL_FILENAME = 'best_kanji_model.pth'
HISTORY_FILENAME = 'training_history.json'
CLASSES_FILENAME = 'classes.json'

# Full file paths
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, MODEL_FILENAME)
HISTORY_SAVE_PATH = os.path.join(OUTPUT_DIR, HISTORY_FILENAME)
CLASSES_SAVE_PATH = os.path.join(OUTPUT_DIR, CLASSES_FILENAME)
MODEL_PATH = MODEL_SAVE_PATH

# Hyperparameters
if QUICK_TEST:
    MAX_CLASSES_LIMIT = 5
    BATCH_SIZE = 2
    NUM_EPOCHS = 1
    OPTUNA_TRIALS = 4
    OPTUNA_EPOCHS = 1
else:
    MAX_CLASSES_LIMIT = 150
    BATCH_SIZE = 64         
    NUM_EPOCHS = 30
    OPTUNA_TRIALS = 15
    OPTUNA_EPOCHS = 15
CHANEL_SIZE = 1 
LEARNING_RATE = 0.00125
WEIGHT_DECAY = 0.01
IMG_SIZE = 64
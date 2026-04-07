import torch
import os

# Paths
DATASET_ROOT = "../../data/synthetic_kanji_dataset"
OUTPUT_DIR = "../output/u-net_training"

# Reproducibility seed
RANDOM_SEED = 42

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Scheduler & Early Stopping
SCHEDULER_PATIENCE = 5
EARLY_STOPPING_PATIENCE = 10
WEIGHT_DECAY = 1e-4

# Model Config
# mobilenet_v3_large is optimized for mobile latency
ENCODER_NAME = "tu-mobilenetv3_large_100" 
ENCODER_WEIGHTS = "imagenet"
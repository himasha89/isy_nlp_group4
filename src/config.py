import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model parameters
MODEL_PARAMS = {
    "embedding_dim": 100,
    "hidden_dim": 128,
    "n_layers": 2,
    "dropout": 0.2,
    "batch_size": 64,
    "learning_rate": 0.001,
    "num_epochs": 10
}

# Text preprocessing parameters
TEXT_PARAMS = {
    "max_length": 100,
    "min_length": 5,
    "min_word_freq": 5
}

# Training parameters
TRAIN_PARAMS = {
    "train_size": 0.7,
    "val_size": 0.15,
    "test_size": 0.15,
    "random_state": 42
}

# Model saving
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = MODEL_DIR / "best_model.pt"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

# Create necessary directories
RAW_DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)
# config.py

import yaml

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

MAX_LENGTH = config['max_length']
THRESHOLD = config['threshold']
BATCH_SIZE = config['batch_size']
LEARNING_RATE = float(config['learning_rate'])
NUM_EPOCHS = config['num_epochs']
MODEL_NAME = config['model_name']
DATASET_NAME = config['dataset_name']
SAVE_DIR = config['save_dir']
TRIPLET_DATASET_PATH = config['triplet_dataset_path']
NUM_SAMPLES = config['num_samples']

import os
import numpy as np

### DATA CONFIGURATION ###

BASE_DATA_FOLDER = "initial_data" # base folder inside "data" where the current data is stored
LABELS_FOLDER = "labels" # folder inside BASE_DATA_FOLDER where labels are stored
SNIPPET_LENGTH_MM = 1

### MODEL CONFIGURATION ###

MODELS_FOLDER = "models"
MODEL_KIND = "autoencoder"
SAVE_MODEL = True

AE_PARAMS = {
    "clf__latent_dim": [30],
    "clf__n_layers": [2],
    "clf__n_epochs": [500],  # [100],
    "clf__lr": [1e-3],  # , 5e-4],
    "clf__batch_size": [512],  # [128]#, 256, 512, 1024],
    "clf__patience": [5],
    "clf__val_split": [0.1],
    # "clf__device": ["cpu"],#"cuda" if torch.cuda.is_available() else "cpu"
}

THRESHOLD_PARAMS = {
    "m": list(range(1, 11)),
    "seg_threshold": list(np.linspace(0.0, 0.1, 21)),
}



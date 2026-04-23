"""
Configuration class with all training parameters.
"""

from pathlib import Path


class Config:
    # --- Paths ---
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    FOLD_DATASET_DIR = PROJECT_ROOT / "fold_dataset"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    CHECKPOINTS_DIR = OUTPUT_DIR / "checkpoints"

    # --- Models ---
    MODEL_NAMES = ["yolo11m.pt", "yolov8m.pt"]
    IMG_SIZE = 640
    NUM_CLASSES = 1

    # --- Loss ---
    CLS = 1.5
    BOX = 7.5
    DFL = 2.0

    # --- Training ---
    EPOCHS = 60
    BATCH_SIZE = 10
    WORKERS = 4

    OPTIMIZER = "AdamW"  
    LR0 = 0.002          
    LRF = 0.01
    COS_LR = True 

    MOMENTUM = 0.937
    WEIGHT_DECAY = 0.0005
    WARMUP_EPOCHS = 3.0
    WARMUP_MOMENTUM = 0.8
    WARMUP_BIAS_LR = 0.1
    AMP = True
    DEVICE = "0"

    # --- Early Stopping ---
    EARLY_STOPPING_PATIENCE = 15

    # --- Augmentations (Ultralytics built-in) ---
    HFLIP_PROB = 0.5
    DEGREES = 15
    TRANSLATE = 0.1
    SCALE = 0.5
    MOSAIC = 1
    HSV_H = 0.015
    HSV_S = 0.7
    HSV_V = 0.4
    ERASING = 0.4

    SHEAR = 1.0   
    PERSPECTIVE = 0.0005

    CLOSE_MOSAIC = 15

    # --- Cross-Validation ---
    NUM_FOLDS = 3
    FOLD_DIR_PATTERN = "fold_{}"
    TRAIN_SUBDIR = "train"
    VAL_SUBDIR = "val"

    VAL_IOU = 0.65

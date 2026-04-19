"""
Main entry point for the training pipeline.
Iterates over models and cross-validation folds.
"""

import traceback
from src.src_train.config import Config
from src.src_train.dataset import validate_fold
from src.src_train.train import train_single_model
from src.src_train.utils import clean_memory

def main():
    config = Config()

    print(" Starting Training ")
    print(f" Models to train: {config.MODEL_NAMES}")
    print(f" Total folds: {config.NUM_FOLDS}")
    print(f" Device: {config.DEVICE}")
    print(f" Dataset root: {config.FOLD_DATASET_DIR}")

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    successful_runs = []

    for fold_idx in range(1, config.NUM_FOLDS + 1):
        fold_dir_name = config.FOLD_DIR_PATTERN.format(fold_idx)
        fold_dir = config.FOLD_DATASET_DIR / fold_dir_name

        if not fold_dir.exists():
            print(f"Directory {fold_dir} not found. Skipping fold {fold_idx}.")
            continue

        train_cnt, val_cnt = validate_fold(fold_dir, config)
        print(f" FOLD {fold_idx} | Images: Train={train_cnt}, Val={val_cnt}")

        if train_cnt == 0 or val_cnt == 0:
            print(f"Empty train or val set in fold {fold_idx}. Skipping.")
            continue

        for model_name in config.MODEL_NAMES:
            print(f"\nStarting training: Model={model_name} | Fold={fold_idx}")
            
            try:
                result = train_single_model(fold_idx, model_name, config)
                successful_runs.append(result)
                print(f" Successfully finished {model_name} on Fold {fold_idx}.")
            except Exception as e:
                print(f" [ERROR] Training failed for {model_name} on Fold {fold_idx}!")
                print(traceback.format_exc())
            finally:
                clean_memory()

    print(" Complete Training Summary")
    if successful_runs:
        print("Successful trainings:")
        for res in successful_runs:
            print(f" Fold {res['fold_idx']} | {res['model_name']} -> {res['best_model_path']}")
    else:
        print(" No models were successfully trained.")

if __name__ == "__main__":
    main()
"""
Dataset utilities - fold handling and YAML config creation for Ultralytics.
"""

import yaml
from pathlib import Path


def create_yaml_config(fold_idx, config):
    """ Create YAML config file for Ultralytics training based on the fold directory structure."""
    fold_dir_name = config.FOLD_DIR_PATTERN.format(fold_idx)
    fold_dir = config.FOLD_DATASET_DIR / fold_dir_name

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yaml_path = config.OUTPUT_DIR / f"data_fold_{fold_idx}.yaml"
    
    yaml_config = {
        "path": str(fold_dir),
        "train": "train/images",
        "val": "val/images",
        "nc": config.NUM_CLASSES,
        "names": ["ship"],
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False)

    return str(yaml_path)


def get_fold_image_paths(fold_dir, subdir):
    """ Get all image paths in a fold subdir"""
    img_dir = fold_dir / subdir / "images"
    if not img_dir.exists():
        return []
    return sorted([
        f for f in img_dir.iterdir() 
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
    ])


def validate_fold(fold_dir, config):
    """ Check if fold has train/val images and return their counts """
    train_imgs = get_fold_image_paths(fold_dir, config.TRAIN_SUBDIR)
    val_imgs = get_fold_image_paths(fold_dir, config.VAL_SUBDIR)
    return len(train_imgs), len(val_imgs)

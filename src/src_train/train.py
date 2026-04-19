"""
Training and validation logic.
"""

from pathlib import Path

from .model import create_model
from .dataset import create_yaml_config
from .utils import clean_memory


def train_single_model(fold_idx, model_name, config):
    """ Train a single model on a single fold. """
    print(f"Training {model_name} on fold {fold_idx}")

    model = create_model(model_name)
    
    checkpoint_dir = config.CHECKPOINTS_DIR / f"fold_{fold_idx}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = create_yaml_config(fold_idx, config)

    results = model.train(
        data=data_yaml,
        epochs=config.EPOCHS,
        imgsz=config.IMG_SIZE,
        batch=config.BATCH_SIZE,
        workers=config.WORKERS,

        cls=config.CLS,             
        box=config.BOX, 
        dfl=config.DFL, 

        optimizer=config.OPTIMIZER,
        lr0=config.LR0,
        lrf=config.LRF,
        cos_lr=config.COS_LR,

        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
        warmup_epochs=config.WARMUP_EPOCHS,
        warmup_momentum=config.WARMUP_MOMENTUM,
        warmup_bias_lr=config.WARMUP_BIAS_LR,

        amp=config.AMP,
        device=config.DEVICE,

        erasing=config.ERASING,
        shear=config.SHEAR, 
        perspective=config.PERSPECTIVE,
        hsv_h=config.HSV_H,
        hsv_s=config.HSV_S,
        hsv_v=config.HSV_V,
        degrees=config.DEGREES,
        translate=config.TRANSLATE,
        scale=config.SCALE,
        fliplr=config.HFLIP_PROB,
        mosaic=config.MOSAIC,
        close_mosaic=config.CLOSE_MOSAIC,

        patience=config.EARLY_STOPPING_PATIENCE,
        val=True,
        project=str(config.OUTPUT_DIR),
        name=f"adv_fold_{fold_idx}_{model_name.replace('.pt', '')}",
        exist_ok=True,
        verbose=True,
        plots=True,
        save=True,
        save_period=-1,
    )

    best_model_path = checkpoint_dir / f"{model_name.replace('.pt', '')}_best.pt"
    
    saved_model_path = Path(config.OUTPUT_DIR) / f"fold_{fold_idx}_{model_name.replace('.pt', '')}" / "weights" / "best.pt"
    if saved_model_path.exists():
        import shutil
        shutil.copy(str(saved_model_path), str(best_model_path))

    if best_model_path.exists():
        print(f"Best model saved to {best_model_path}")

    del model
    clean_memory()

    return {
        "fold_idx": fold_idx,
        "model_name": model_name,
        "best_model_path": str(best_model_path),
        "results": results,
    }

"""
Скрипт обучения компактной модели-ученика YOLOv8n.
Обучает легкую модель на дистиллированном датасете (псевдо-метки от ансамбля) + исходных данных
с целью минимизации задержки инференса. Использует агрессивную аугментацию (включая erasing)
для повышения робастности, валидирует только на Ground Truth разметке.
"""

import gc
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"                           
DATA_YAML = "fold_dataset/data.yaml"                  
CLEAN_VAL_IMAGES = "fold_dataset/test_images/images"  
OUTPUT_DIR = "D:/LabsMIET/Detection-ship/outputs/student_distilled"               

EPOCHS = 80
BATCH_SIZE = 32           
IMG_SIZE = 640
DEVICE = "0" if torch.cuda.is_available() else "cpu"
WORKERS = 4

MOSAIC = 1.0              
SCALE = 0.5                
TRANSLATE = 0.1         
DEGREES = 10.0            
FLIP_LR = 0.5             
FLIP_UD = 0.0             
SHEAR = 0.0                
ERASING = 0.4             
PERSPECTIVE = 0.0005

HSV_H = 0.015              
HSV_S = 0.7         
HSV_V = 0.4    

CLOSE_MOSAIC = 15          
LR0 = 0.01                 
WEIGHT_DECAY = 0.0005



def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def fix_validation_path(yaml_path, clean_val_dir):
    """Направляет валидацию Ученика только на Ground Truth разметку."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    data['val'] = str(Path(clean_val_dir).absolute())
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f" Валидация настроена на чистые данные: {clean_val_dir}")


def train_student():
    print(f" Обучение Ученика: {MODEL_NAME}")

    fix_validation_path(DATA_YAML, CLEAN_VAL_IMAGES)

    model = YOLO(MODEL_NAME)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        workers=WORKERS,
        project=OUTPUT_DIR,
        name="adv_yolov8n_student",
        exist_ok=True,
        
        lr0=LR0,
        weight_decay=WEIGHT_DECAY,
        
        mosaic=MOSAIC,
        scale=SCALE,
        translate=TRANSLATE,
        degrees=DEGREES,
        fliplr=FLIP_LR,
        flipud=FLIP_UD,
        shear=SHEAR,
        mixup=0.0,        
        erasing=ERASING,  
        perspective=PERSPECTIVE,

        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        
        close_mosaic=CLOSE_MOSAIC,
        
        val=True,
        save=True,
        plots=True,
        verbose=True
    )

    print("Обучение Ученика завершено")
    print(f" Лучшие веса сохранены в: {Path(OUTPUT_DIR) / 'adv_yolov8n_student' / 'weights' / 'best.pt'}")

    del model
    clean_memory()

if __name__ == "__main__":
    train_student()
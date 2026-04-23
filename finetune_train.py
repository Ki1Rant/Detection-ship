"""
Скрипт дообучения ансамбля на выбранных фолдах.
Выполняет специализированное дообучение моделей YOLOv8m и YOLO11m с увеличением разрешения до 1024×1024.
Использует расширенную аугментацию (mosaic, flip, scale), ранную остановку и оптимизацию памяти.
Сохраняет дообученные модели для последующей дистилляции и финального инференса.
"""

import os
import gc
import torch
from ultralytics import YOLO


BASE_WEIGHTS_DIR = "outputs/adv_checkpoints"
TARGET_FOLDS = [1, 2, 3]                  
MODEL_TYPES = ["yolo11m", "yolo11m"]     

DATA_YAML = "dataset_finetune/data.yaml"           
PROJECT_NAME = "D:/LabsMIET/Detection-ship/outputs/finetune"           

IMG_SIZE = 1024        
EPOCHS = 50  
PATIENCE = 10          
BATCH_SIZE = 8         
DEVICE = 0             

LR_INITIAL = 0.001     
FREEZE_LAYERS = 10    


MOSAIC = 1.0           
SCALE = 0.5            
FLIP_LR = 0.5          
CLOSE_MOSAIC = 10      

def clear_memory():
    """Принудительная очистка оперативной и видеопамяти."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def main():
    print(" Дообучение ")

    if not os.path.exists(DATA_YAML):
        print(f"[КРИТИЧЕСКАЯ ОШИБКА] Файл датасета {DATA_YAML} не найден!")
        return

    total_runs = len(TARGET_FOLDS) * len(MODEL_TYPES)
    current_run = 1

    for fold in TARGET_FOLDS:
        for model_type in MODEL_TYPES:
            print(f" Обучение {current_run}/{total_runs}: {model_type.upper()} | ФОЛД {fold} ")

            # Формируем пути
            weights_path = os.path.join(BASE_WEIGHTS_DIR, f"fold_{fold}", f"{model_type}_best.pt")
            exp_name = f"{model_type}_fold{fold}_img{IMG_SIZE}"

            if not os.path.exists(weights_path):
                print(f"Веса не найдены: {weights_path}")
                current_run += 1
                continue

            try:
                print(f"[ИНФО] Загрузка весов: {weights_path}")
                model = YOLO(weights_path)

                # Запуск обучения
                model.train(
                    data=DATA_YAML,
                    project=PROJECT_NAME,
                    name=exp_name,
                    
                    epochs=EPOCHS,
                    imgsz=IMG_SIZE,
                    batch=BATCH_SIZE,
                    patience=PATIENCE,
                    device=DEVICE,
                    
                    lr0=LR_INITIAL,
                    freeze=FREEZE_LAYERS,
                    
                    mosaic=MOSAIC,
                    scale=SCALE,
                    fliplr=FLIP_LR,
                    close_mosaic=CLOSE_MOSAIC,
                    
                    optimizer='auto',
                    exist_ok=True,
                    plots=True,
                    val=True
                )
                
                print(f"\nМодель {model_type} (fold {fold}) обучена")
                print(f"Результаты: {os.path.join(PROJECT_NAME, exp_name)}")

            except Exception as e:
                print(f"\n[Ошибка во время обучения] {model_type} (fold {fold}):")
                print(e)
                print("Переход к следующей модели...")

            finally:

                print("Очистка памяти")
                if 'model' in locals():
                    del model  
                clear_memory() 
            
            current_run += 1

    print(" Обучение завершено ")

if __name__ == "__main__":
    clear_memory()
    main()
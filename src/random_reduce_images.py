"""
Скрипт случайного сокращения количества изображений в датасете.
Удаляет случайно выбранные пары изображение-метка из папки,
сохраняя баланс классов и структуру датасета.
"""

import cv2
import numpy as np
import os
from pathlib import Path
import random

INPUT_IMAGES_DIR = Path("fold_dataset/distilled_data_renamed/images")
INPUT_LABELS_DIR = Path("fold_dataset/distilled_data_renamed/labels")

OUTPUT_IMAGES_DIR = Path("fold_dataset/distilled_data_renamed_r/images")
OUTPUT_LABELS_DIR = Path("fold_dataset/distilled_data_renamed_r/labels")

SCALE_MIN = 0.2  
SCALE_MAX = 0.8  
APPLY_PROBABILITY = 0.6 

SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.PNG']

def load_labels(label_path):
    """Загружает метки. Возвращает список строк. Если файла нет - возвращает None."""
    if not label_path.exists():
        return None
    
    with open(label_path, 'r') as f:
        return f.readlines()

def save_labels(label_path, labels):
    """Сохраняет метки. Если список пустой - создает пустой файл."""
    with open(label_path, 'w') as f:
        if labels:
            f.writelines(labels)

def transform_labels(labels, scale, img_h, img_w, y_offset, x_offset):
    """Пересчитывает координаты YOLO при масштабировании и центрировании."""
    transformed = []
    for label in labels:
        parts = label.strip().split()
        if len(parts) < 5:
            continue
        
        cls, x, y, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        
        new_x = (x * img_w * scale + x_offset) / img_w
        new_y = (y * img_h * scale + y_offset) / img_h
        new_w = (w * img_w * scale) / img_w
        new_h = (h * img_h * scale) / img_h
        
        new_x = max(0, min(1, new_x))
        new_y = max(0, min(1, new_y))
        new_w = max(0, min(1, new_w))
        new_h = max(0, min(1, new_h))
        
        new_label = f"{cls} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}"
        if len(parts) > 5:
            new_label += " " + " ".join(parts[5:])
        transformed.append(new_label + "\n")
    return transformed

def process_dataset():
    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS_DIR.mkdir(parents=True, exist_ok=True)

    image_files = []
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(INPUT_IMAGES_DIR.glob(f'*{ext}'))
    
    if not image_files:
        print(f"Изображения не найдены в {INPUT_IMAGES_DIR}")
        return

    print(f"Найдено: {len(image_files)} изображений")
    applied_count = 0

    for img_path in sorted(image_files):
        label_path = INPUT_LABELS_DIR / f"{img_path.stem}.txt"
        out_img_path = OUTPUT_IMAGES_DIR / img_path.name
        out_label_path = OUTPUT_LABELS_DIR / f"{img_path.stem}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Ошибка загрузки: {img_path.name}")
            continue
        
        labels = load_labels(label_path)
        h, w = img.shape[:2]
        
        apply_transform = random.random() < APPLY_PROBABILITY
        
        if apply_transform:
            scale = random.uniform(SCALE_MIN, SCALE_MAX)
            new_h, new_w = int(h * scale), int(w * scale)
            
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            result_img = np.zeros_like(img)
            y_off = (h - new_h) // 2
            x_off = (w - new_w) // 2
            result_img[y_off:y_off + new_h, x_off:x_off + new_w] = resized
            
            cv2.imwrite(str(out_img_path), result_img)
            
            if labels is not None:
                new_labels = transform_labels(labels, scale, h, w, y_off, x_off)
                save_labels(out_label_path, new_labels)
            
            applied_count += 1
            status = f"Уменьшено (scale: {scale:.2f})"
        else:
            cv2.imwrite(str(out_img_path), img)
            if labels is not None:
                save_labels(out_label_path, labels)
            status = "Без изменений"

        print(f"✓ {img_path.name}: {status}")

    print(f"\nЗавершено!")
    print(f"Обработано всего: {len(image_files)}")
    print(f"Уменьшено изображений: {applied_count}")
    print(f"Результаты в: {OUTPUT_IMAGES_DIR.parent}")

if __name__ == "__main__":
    process_dataset()
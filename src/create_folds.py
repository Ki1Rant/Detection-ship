"""
Скрипт для разбиения датасета на тестовую выборку и K-fold кросс-валидационные сеты.
Скрипт считывает изображения и соответствующие им метки, распределяет их случайным образом,
обеспечивая отсутствие пересечений между тестовым набором и фолдами.
"""

import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import KFold
import numpy as np

SEED = 42
TOTAL_IMAGES = 4885
TEST_COUNT = 400
NUM_FOLDS = 5
TRAIN_RATIO = 0.8

SRC_DIR = Path(__file__).resolve().parent.parent / "dataset_clean_r"
SRC_IMAGES = SRC_DIR / "images"
SRC_LABELS = SRC_DIR / "labels"
DEST_DIR = Path(__file__).resolve().parent.parent / "fold_dataset_r"


def get_image_names():
    """Возвращает отсортированный список базовых имен файлов изображений (без расширения)."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = sorted([f for f in os.listdir(SRC_IMAGES) if Path(f).suffix.lower() in extensions])
    return [Path(f).stem for f in images]


def verify_pairs(names):
    """Проверяет наличие соответствующих файлов меток для каждого изображения."""
    missing = []
    for name in names:
        img_found = False
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            if (SRC_IMAGES / f"{name}{ext}").exists():
                img_found = True
                break
        if not img_found:
            missing.append(f"Изображение отсутствует: {name}")
        if not (SRC_LABELS / f"{name}.txt").exists():
            missing.append(f"Метка отсутствует: {name}")
    if missing:
        raise FileNotFoundError(f"Отсутствующие файлы:\n" + "\n".join(missing[:10]))
    print(f"  Верифицировано {len(names)} пар изображение-метка — все файлы на месте.")


def find_image_path(name):
    """Находит файл изображения с любым поддерживаемым расширением."""
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
        p = SRC_IMAGES / f"{name}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Изображение не найдено для: {name}")


def copy_pairs(names, dest_images_dir, dest_labels_dir):
    """Копирует пары изображение-метка в целевые директории."""
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    dest_labels_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        img_src = find_image_path(name)
        lbl_src = SRC_LABELS / f"{name}.txt"
        shutil.copy2(img_src, dest_images_dir / img_src.name)
        shutil.copy2(lbl_src, dest_labels_dir / f"{name}.txt")


def main():
    random.seed(SEED)

    if DEST_DIR.exists():
        shutil.rmtree(DEST_DIR)

    all_names = get_image_names()
    print(f"Всего найдено изображений: {len(all_names)}")
    random.shuffle(all_names)

    test_names = all_names[:TEST_COUNT]
    remaining = all_names[TEST_COUNT:]
    print(f"Тестовая выборка: {len(test_names)} изображений")
    print(f"Осталось для фолдов: {len(remaining)} изображений")

    print("\nСоздание тестовой директории")
    verify_pairs(test_names)
    copy_pairs(
        test_names,
        DEST_DIR / "test_images" / "images",
        DEST_DIR / "test_images" / "labels",
    )

    print(f"\nСоздание {NUM_FOLDS}-fold разбиений ({len(remaining)} изображений всего)")
    
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    remaining_np = np.array(remaining)
    
    for i, (train_idx, val_idx) in enumerate(kf.split(remaining_np), start=1):
        print(f"\nСоздание fold_{i}...")
        
        train_names = remaining_np[train_idx].tolist()
        val_names = remaining_np[val_idx].tolist()
        
        print(f"  Обучение: {len(train_names)}, Валидация: {len(val_names)}")

        assert len(set(train_names) & set(val_names)) == 0, \
            f"Утечка данных в fold_{i}: пересечение обучения и валидации!"

        verify_pairs(train_names + val_names)

        copy_pairs(
            train_names,
            DEST_DIR / f"fold_{i}" / "train" / "images",
            DEST_DIR / f"fold_{i}" / "train" / "labels",
        )
        copy_pairs(
            val_names,
            DEST_DIR / f"fold_{i}" / "val" / "images",
            DEST_DIR / f"fold_{i}" / "val" / "labels",
        )

    print("\nФинальная проверка на утечку данных")
    test_set = set(test_names)
    for i in range(1, NUM_FOLDS + 1):
        fold_set = set()
        train_dir = DEST_DIR / f"fold_{i}" / "train" / "labels"
        val_dir = DEST_DIR / f"fold_{i}" / "val" / "labels"
        for f in os.listdir(train_dir):
            fold_set.add(Path(f).stem)
        for f in os.listdir(val_dir):
            fold_set.add(Path(f).stem)

        assert len(test_set & fold_set) == 0, f"Утечка: пересечение test и fold_{i}!"
        print(f"  fold_{i}: {len(fold_set)} изображений — утечек с тестовым набором нет")

    print("\nУтечек данных не обнаружено.")
    print(f"\nГотово. Датасет создан: {DEST_DIR}")
    print(f"  test_images/: {len(test_names)}")
    for i in range(1, NUM_FOLDS + 1):
        train_count = int(len(remaining) * TRAIN_RATIO)
        val_count = len(remaining) - train_count
        print(f"  fold_{i}/: всего {len(remaining)} (обучение: {train_count}, валидация: {val_count})")


if __name__ == "__main__":
    main()
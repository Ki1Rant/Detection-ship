"""
Скрипт удаления дубликатов из датасета.
Использует PHash (перцептивный хеш) для поиска похожих/идентичных изображений
и удаления дубликатов с сохранением исходных пар изображение-метка.
"""

import os
from pathlib import Path
from imagededup.methods import PHash

DATASET_ROOT = Path(__file__).resolve().parent.parent / "fold_dataset/distilled_data_renamed"

def collect_all_images(images_dir: Path) -> list[Path]:
    """Собирает все изображения из конкретной папки images."""
    all_images = []
    if not images_dir.exists():
        print(f"Ошибка: Директория {images_dir} не найдена!")
        return []
    
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        all_images.extend(images_dir.glob(ext))
    return all_images

def find_and_remove_duplicates():
    images_dir = DATASET_ROOT / "images"
    labels_dir = DATASET_ROOT / "labels"

    all_images = collect_all_images(images_dir)
    if not all_images:
        print(f"Изображения не найдены в: {images_dir}")
        return

    print(f"Всего изображений для анализа: {len(all_images)}")

    phasher = PHash()
    encodings = {}
    print("Вычисление хешей (это может занять время)...")
    for img_path in all_images:
        try:
            img_hash = phasher.encode_image(str(img_path))
            if img_hash:
                encodings[str(img_path)] = img_hash
        except Exception as e:
            print(f"Ошибка при обработке {img_path.name}: {e}")

    files_to_remove = phasher.find_duplicates_to_remove(
        encoding_map=encodings,
        max_distance_threshold=4 
    )

    if not files_to_remove:
        print("Дубликаты не обнаружены!")
        return

    print(f"Найдено дубликатов для удаления: {len(files_to_remove)}")

    removed_count = 0
    for file_path_str in files_to_remove:
        img_path = Path(file_path_str)
        
        if img_path.exists():
            label_path = labels_dir / (img_path.stem + ".txt")
            
            if label_path.exists():
                label_path.unlink()
            
            img_path.unlink()
            removed_count += 1

    print(f"Очистка завершена")
    print(f"  Удалено дубликатов: {removed_count}")
    print(f"  Осталось изображений: {len(all_images) - removed_count}")

if __name__ == "__main__":
    find_and_remove_duplicates()
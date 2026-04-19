"""
Скрипт для создания датасета дистилляции (Псевдо-разметка ансамблем).
Прогоняет неразмеченные картинки через ансамбль, объединяет предсказания (WBF),
и формирует готовую YOLO-директорию с images, labels и dataset.yaml.
"""

import shutil
import torch
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion

MODEL_WEIGHTS = [
    "outputs/adv_checkpoints/fold_1/yolo11m_best.pt",
    "outputs/adv_checkpoints/fold_2/yolo11m_best.pt",
    "outputs/adv_checkpoints/fold_3/yolo11m_best.pt",
    "outputs/adv_checkpoints/fold_1/yolov8m_best.pt",
    "outputs/adv_checkpoints/fold_2/yolov8m_best.pt",
    "outputs/adv_checkpoints/fold_3/yolov8m_best.pt"
]

INPUT_IMAGES_DIR = "fold_dataset/distilled_data_renamed/train/images" 
OUTPUT_DATASET_DIR = "fold_dataset/distilled_data_renamed/train/labels" 

IMG_SIZE = 640
DEVICE = "0" if torch.cuda.is_available() else "cpu"

WBF_SKIP_CONF = 0.10     
WBF_IOU_THRES = 0.35     
FINAL_CONF_THRES = 0.30 

COPY_IMAGES = False       # Копировать ли физически картинки в новую папку (True - да, False - только txt)


def main():
    out_dir = Path(OUTPUT_DATASET_DIR)
    out_images_dir = out_dir / "images" / "train"
    out_labels_dir = out_dir 
    
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    print(" Загрузка моделей ансамбля")
    models = [YOLO(w) for w in MODEL_WEIGHTS]

    # Ищем все картинки
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(Path(INPUT_IMAGES_DIR).rglob(ext))
    
    if not image_paths:
        print(f" Картинки не найдены в {INPUT_IMAGES_DIR}")
        return

    print(f" Найдено неразмеченных изображений: {len(image_paths)}")
    print(f" Старт генерации псевдо-разметки (Дистилляция)")

    # Основной цикл с прогресс-баром
    for img_path in tqdm(image_paths, desc="Обработка ансамблем"):
        wbf_boxes_list = []
        wbf_scores_list = []
        wbf_labels_list = []

        # Прогон через все модели
        for model in models:
            results = model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=WBF_SKIP_CONF, device=DEVICE, verbose=False)[0]
            
            boxes = results.boxes.xyxyn.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            labels = results.boxes.cls.cpu().numpy()

            if len(boxes) == 0:
                wbf_boxes_list.append([])
                wbf_scores_list.append([])
                wbf_labels_list.append([])
                continue

            wbf_boxes_list.append(boxes.tolist())
            wbf_scores_list.append(scores.tolist())
            wbf_labels_list.append(labels.tolist())

        # Применяем WBF
        if not any(wbf_boxes_list): # Если все модели промолчали
            f_boxes, f_scores, f_labels = [], [], []
        else:
            f_boxes, f_scores, f_labels = weighted_boxes_fusion(
                wbf_boxes_list, wbf_scores_list, wbf_labels_list,
                weights=[1] * len(models), 
                iou_thr=WBF_IOU_THRES, 
                skip_box_thr=WBF_SKIP_CONF
            )

            keep_indices = [i for i, score in enumerate(f_scores) if score >= FINAL_CONF_THRES]
            f_boxes = [f_boxes[i] for i in keep_indices]
            f_scores = [f_scores[i] for i in keep_indices]
            f_labels = [f_labels[i] for i in keep_indices]

        # Сохранение разметки .txt
        label_file = out_labels_dir / f"{img_path.stem}.txt"
        with open(label_file, 'w') as f:
            for box, score, label in zip(f_boxes, f_scores, f_labels):
                nx1, ny1, nx2, ny2 = box
                
                nx1, ny1 = max(0.0, nx1), max(0.0, ny1)
                nx2, ny2 = min(1.0, nx2), min(1.0, ny2)

                cx = (nx1 + nx2) / 2.0
                cy = (ny1 + ny2) / 2.0
                w = nx2 - nx1
                h = ny2 - ny1
                
                f.write(f"{int(label)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

        if COPY_IMAGES:
            dest_img = out_images_dir / img_path.name
            if not dest_img.exists():
                shutil.copy2(img_path, dest_img)

    print("\n Готово")
    print(f" Путь: {out_dir.absolute()}")

if __name__ == "__main__":
    main()
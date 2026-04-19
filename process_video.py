"""
Скрипт для покадровой обработки видео батчами.
Использует движок из detector.py.

Поддерживает обработку нескольких видеофайлов.
"""

import cv2
import os
from tqdm import tqdm
from src.detector import ShipDetector


INPUT_VIDEO_PATHS = [
    "fold_dataset/video_1.mp4",
    "fold_dataset/video_2.mp4",
    "fold_dataset/video_3.mp4",
    "fold_dataset/video_5.mp4",
    "fold_dataset/video_9.mp4",
    "fold_dataset/video_10.mp4",
]

OUTPUT_DIR = "outputs"
OUTPUT_BASENAME = "result_video"

BATCH_SIZE = 16  

# Настройки отрисовки
BOX_COLOR = (0, 255, 0)   
TEXT_COLOR = (0, 0, 0)    
THICKNESS = 7            

def draw_boxes(frame, detections):
    """Отрисовывает боксы на одном кадре."""
    h, w = frame.shape[:2]
    
    for det in detections:
        nx1, ny1, nx2, ny2 = det["box"]
        x1, y1 = int(nx1 * w), int(ny1 * h)
        x2, y2 = int(nx2 * w), int(ny2 * h)
        score = det["score"]
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, THICKNESS)
        
        label_text = f"Ship: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw, y1), BOX_COLOR, -1)
        
        cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
        
    return frame

def process_single_video(input_path, output_path, detector):
    """Обрабатывает одно видео и сохраняет результат."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео: {input_path}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f" Видео загружено: {width}x{height} | {fps} FPS | Всего кадров: {total_frames}")
    print(f" Старт обработки (Батч = {BATCH_SIZE})...")

    frames_buffer = []
    with tqdm(total=total_frames, desc=f"Обработка {os.path.basename(input_path)}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_buffer.append(frame)

            if len(frames_buffer) == BATCH_SIZE:
                batch_results = detector.predict_batch(frames_buffer)
                for f, dets in zip(frames_buffer, batch_results):
                    processed_frame = draw_boxes(f, dets)
                    out.write(processed_frame)
                frames_buffer = []
                pbar.update(BATCH_SIZE)

        if frames_buffer:
            batch_results = detector.predict_batch(frames_buffer)
            for f, dets in zip(frames_buffer, batch_results):
                processed_frame = draw_boxes(f, dets)
                out.write(processed_frame)
            pbar.update(len(frames_buffer))

    cap.release()
    out.release()
    print(f"Готово: {output_path}\n")
    return True

def main():
    if not INPUT_VIDEO_PATHS:
        print("Список видео пуст. Добавьте пути в INPUT_VIDEO_PATHS.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Инициализация детектора")
    detector = ShipDetector()

    print(f"\nНайдено видеофайлов: {len(INPUT_VIDEO_PATHS)}")

    for idx, video_path in enumerate(INPUT_VIDEO_PATHS, start=1):
        if not os.path.exists(video_path):
            print(f"Пропуск: файл не найден -> {video_path}")
            continue

        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{OUTPUT_BASENAME}_{idx}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        print(f"\n Обработка видео {idx}/{len(INPUT_VIDEO_PATHS)}: {video_path}")
        success = process_single_video(video_path, output_path, detector)
        if not success:
            print(f" Не удалось обработать видео: {video_path}")

    print(" \n Все видео обработаны!")
    print(f" Результаты сохранены в папке: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
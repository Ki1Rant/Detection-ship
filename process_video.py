"""
Скрипт инференса на видеофайлах с трекингом объектов.
Использует ансамбль дообученных моделей (YOLO11m/YOLOv8m) с SAHI (Sliced Aided Hyper Inference)
для детекции кораблей на больших кадрах. Реализует IoU-based трекинг объектов между фреймами
с автоматической раскраской треков и FPS-оптимизацией через батчевую обработку.
"""

import cv2
import os
import random
from tqdm import tqdm
from src.detector import ShipDetector

INPUT_VIDEO_PATHS = [
    "fold_dataset/video_2.mp4",
    "fold_dataset/video_5.mp4"
]
OUTPUT_DIR = "outputs"
OUTPUT_BASENAME = "ensemble_sahi_result"
BATCH_SIZE = 8  

TRACK_COLORS = {}

def get_color(track_id):
    if track_id not in TRACK_COLORS:
        TRACK_COLORS[track_id] = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    return TRACK_COLORS[track_id]

def draw_boxes(frame, detections):
    h, w = frame.shape[:2]
    for det in detections:
        nx1, ny1, nx2, ny2 = det["box"]
        x1, y1, x2, y2 = int(nx1*w), int(ny1*h), int(nx2*w), int(ny2*h)
        tid, conf = det["track_id"], det["score"]
        
        color = get_color(tid)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        label = f"ID:{tid} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def process_single_video(input_path, output_path, detector):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): return False
    
    width, height, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
    total_frames = int(cap.get(7))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    detector.reset_tracker()
    frames_buffer = []

    with tqdm(total=total_frames, desc=f"Ensemble: {os.path.basename(input_path)}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames_buffer.append(frame)

            if len(frames_buffer) == BATCH_SIZE:
                batch_results = detector.predict_batch(frames_buffer)
                for f, dets in zip(frames_buffer, batch_results):
                    out.write(draw_boxes(f, dets))
                frames_buffer = []
                pbar.update(BATCH_SIZE)

        if frames_buffer:
            batch_results = detector.predict_batch(frames_buffer)
            for f, dets in zip(frames_buffer, batch_results):
                out.write(draw_boxes(f, dets))
            pbar.update(len(frames_buffer))

    cap.release()
    out.release()
    return True

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    detector = ShipDetector()
    for idx, video_path in enumerate(INPUT_VIDEO_PATHS, start=1):
        if not os.path.exists(video_path): continue
        out_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}_{idx}.mp4")
        process_single_video(video_path, out_path, detector)

if __name__ == "__main__":
    main()
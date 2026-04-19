"""
Движок инференса для детекции кораблей с поддержкой SAHI.
Используется NMS.
"""

import torch
import torchvision
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction 

USE_SAHI = False  

MODEL_WEIGHTS = [
    "D:/LabsMIET/Detection-ship/outputs/student_distilled/yolov8n_student/weights/best.pt",
]

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

SAHI_SLICE_SIZE = 640
SAHI_OVERLAP = 0.2    

PRE_NMS_CONF = 0.3      
NMS_IOU_THRES = 0.4     
FINAL_CONF_THRES = 0.3  


class ShipDetector:
    def __init__(self, use_sahi=USE_SAHI):
        self.use_sahi = use_sahi
        
        status = "Включен SAHI " if self.use_sahi else "Выключен SAHI"
        
        self.model = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',
            model_path=MODEL_WEIGHTS[0],
            confidence_threshold=PRE_NMS_CONF,
            device=DEVICE
        )
        print(" Модель загружена")

    def predict_batch(self, frames_batch):
        """
        Принимает батч картинок из OpenCV.
        Возвращает результаты предсказаний для каждого кадра.
        """
        batch_size = len(frames_batch)
        final_results = []

        for frame in frames_batch:
            img_h, img_w = frame.shape[:2]

            if self.use_sahi:
                result = get_sliced_prediction(
                    frame,
                    self.model,
                    slice_height=SAHI_SLICE_SIZE,
                    slice_width=SAHI_SLICE_SIZE,
                    overlap_height_ratio=SAHI_OVERLAP,
                    overlap_width_ratio=SAHI_OVERLAP,
                    postprocess_match_metric="IOU", 
                    postprocess_match_threshold=0.5, 
                    verbose=0
                )
            else:
                result = get_prediction(
                    frame,
                    self.model,
                    verbose=0
                )
            
            boxes, scores, labels = [], [], []
            
            for obj in result.object_prediction_list:
                x1 = max(0.0, obj.bbox.minx / img_w)
                y1 = max(0.0, obj.bbox.miny / img_h)
                x2 = min(1.0, obj.bbox.maxx / img_w)
                y2 = min(1.0, obj.bbox.maxy / img_h)

                boxes.append([x1, y1, x2, y2])
                scores.append(obj.score.value)
                labels.append(obj.category.id)

            if len(boxes) == 0:
                final_results.append([])
                continue

            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(scores, dtype=torch.float32)

            keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, NMS_IOU_THRES)

            frame_res = []
            for idx in keep_indices:
                score = scores[idx]
                if score >= FINAL_CONF_THRES:
                    frame_res.append({
                        "box": boxes[idx],
                        "score": float(score),
                        "label": int(labels[idx])
                    })
            
            final_results.append(frame_res)

        return final_results
"""
Класс ShipDetector для инференса с ансамблем моделей и трекингом объектов.
Использует SAHI для детекции на больших кадрах путем разрезания на сегменты (640×640 с перекрытием 25%).
Реализует IoU-based трекинг с параметрами гибкости, фильтрацией вложенных боксов (мачты/надстройки)
и сглаживанием координат объектов между фреймами.
"""

import torch
import numpy as np
import torchvision
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

MODEL_PATHS = [
    "D:\\LabsMIET\\Detection-ship\\outputs\\finetune\\yolo11m_fold1_img1024\\weights\\best.pt",
    "D:\\LabsMIET\\Detection-ship\\outputs\\finetune\\yolo11m_fold2_img1024\\weights\\best.pt",
    "D:\\LabsMIET\\Detection-ship\\outputs\\finetune\\yolo11m_fold3_img1024\\weights\\best.pt",
    "D:\\LabsMIET\\Detection-ship\\outputs\\finetune\\yolov8m_fold1_img1024\\weights\\best.pt",
    "D:\\LabsMIET\\Detection-ship\\outputs\\finetune\\yolov8m_fold2_img1024\\weights\\best.pt",
    "D:\\LabsMIET\\Detection-ship\\outputs\\finetune\\yolov8m_fold3_img1024\\weights\\best.pt"
]
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

SLICE_SIZE = 640
OVERLAP = 0.25
CONF_THRES = 0.45            

ENSEMBLE_IOU_THRES = 0.5    
NESTED_IOU_THRES = 0.85      
MIN_AREA_RATIO = 0.0005      

MIN_HITS = 3                
MAX_AGE = 12                
SMOOTHING_ALPHA = 0.3        


class ShipDetector:
    def __init__(self):
        print(f"Загрузка ансамбля из {len(MODEL_PATHS)} моделей...")
        self.models = []
        for path in MODEL_PATHS:
            m = AutoDetectionModel.from_pretrained(
                model_type='ultralytics',
                model_path=path,
                confidence_threshold=CONF_THRES,
                device=DEVICE
            )
            self.models.append(m)
        
        self.trackers = {}  
        self.next_id = 1

    def reset_tracker(self):
        """Полный сброс трекера между видео."""
        self.trackers = {}
        self.next_id = 1

    def _get_iou(self, boxA, boxB):
        """Расчет IoU между двумя боксами [x1, y1, x2, y2]."""
        xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
        xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter / float(areaA + areaB - inter + 1e-6)

    def _filter_nested_boxes(self, detections):
        """Удаляет боксы, которые являются частями (мачты, надстройки) большого корабля."""
        if not detections: return []
        # Сортировка по площади (сначала большие)
        dets = sorted(detections, key=lambda x: (x['box'][2]-x['box'][0]) * (x['box'][3]-x['box'][1]), reverse=True)
        keep = []
        for det_inner in dets:
            is_inside = False
            ix1, iy1, ix2, iy2 = det_inner["box"]
            inner_area = (ix2 - ix1) * (iy2 - iy1)
            for det_outer in keep:
                ox1, oy1, ox2, oy2 = det_outer["box"]
                xx1, yy1 = max(ix1, ox1), max(iy1, oy1)
                xx2, yy2 = min(ix2, ox2), min(iy2, oy2)
                inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                if inter / (inner_area + 1e-6) > NESTED_IOU_THRES:
                    is_inside = True; break
            if not is_inside: keep.append(det_inner)
        return keep

    def _update_tracker(self, current_dets):
        """Привязка ID, сглаживание координат и фильтрация мерцания воды."""
        updated_results = []
        for tid in self.trackers: self.trackers[tid]["age"] += 1

        matched_indices = set()
        for tid, track in self.trackers.items():
            best_iou, best_idx = 0, -1
            for i, det in enumerate(current_dets):
                if i in matched_indices: continue
                iou = self._get_iou(track["box"], det["box"])
                if iou > 0.2 and iou > best_iou:
                    best_iou, best_idx = iou, i
            
            if best_idx != -1:
                old_b, new_b = np.array(track["box"]), np.array(current_dets[best_idx]["box"])
                smoothed_box = (old_b * (1 - SMOOTHING_ALPHA) + new_b * SMOOTHING_ALPHA).tolist()
                
                self.trackers[tid].update({
                    "box": smoothed_box,
                    "hits": min(track["hits"] + 1, 20), 
                    "age": 0,
                    "score": current_dets[best_idx]["score"]
                })
                matched_indices.add(best_idx)

        # Создание новых треков
        for i, det in enumerate(current_dets):
            if i not in matched_indices:
                self.trackers[self.next_id] = {
                    "box": det["box"], "hits": 1, "age": 0, "score": det["score"]
                }
                self.next_id += 1

        # Фильтрация и выдача результата
        new_trackers = {}
        for tid, track in self.trackers.items():
            if track["age"] < MAX_AGE:
                new_trackers[tid] = track
                if track["hits"] >= MIN_HITS:
                    updated_results.append({
                        "box": track["box"],
                        "track_id": tid,
                        "score": track["score"]
                    })
        self.trackers = new_trackers
        return updated_results

    def predict_batch(self, frames_batch):
        """Основной цикл инференса: Ансамбль -> SAHI -> NMS -> Фильтры -> Трекер."""
        final_results = []

        for frame in frames_batch:
            img_h, img_w = frame.shape[:2]
            all_boxes, all_scores, all_labels = [], [], []

            # 1. Инференс каждой модели в ансамбле через SAHI
            for model in self.models:
                result = get_sliced_prediction(
                    frame, model,
                    slice_height=SLICE_SIZE, slice_width=SLICE_SIZE,
                    overlap_height_ratio=OVERLAP, overlap_width_ratio=OVERLAP,
                    perform_standard_pred=True, 
                    postprocess_type="NMM", verbose=0
                )
                for obj in result.object_prediction_list:
                    all_boxes.append([obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy])
                    all_scores.append(obj.score.value)
                    all_labels.append(obj.category.id)

            if not all_boxes:
                self._update_tracker([])
                final_results.append([])
                continue

            # Объединение предсказаний от разных моделей 
            boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
            scores_t = torch.tensor(all_scores, dtype=torch.float32)
            keep_idx = torchvision.ops.nms(boxes_t, scores_t, ENSEMBLE_IOU_THRES)

            current_frame_dets = []
            for idx in keep_idx:
                b = all_boxes[idx]
                area = (b[2]-b[0]) * (b[3]-b[1]) / (img_w * img_h)
                if area < MIN_AREA_RATIO: continue

                current_frame_dets.append({
                    "box": [b[0]/img_w, b[1]/img_h, b[2]/img_w, b[3]/img_h],
                    "score": float(all_scores[idx]),
                    "label": int(all_labels[idx])
                })

            # Геометрическая фильтрация 
            filtered_dets = self._filter_nested_boxes(current_frame_dets)
            
            # Трекинг и временная стабилизация
            tracked_dets = self._update_tracker(filtered_dets)
            
            final_results.append(tracked_dets)

        return final_results
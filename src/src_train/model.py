"""
Model creation utilities.
"""

from ultralytics import YOLO


def create_model(model_name):
    """Create/load a YOLO model."""
    return YOLO(model_name)

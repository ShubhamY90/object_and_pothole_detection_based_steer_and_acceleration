"""
perception/pothole_detector.py — Detects potholes on the road.

Currently acts as a stub / placeholder so the architecture is fully prepared.
You can replace the contents of detect() with inference from a fine-tuned
YOLOv8 model (e.g. `yolov8_pothole.pt`) when you are ready.
"""

import numpy as np

class PotholeDetector:
    def __init__(self, model_path: str = "pothole_yolov8.pt"):
        self.model_path = model_path
        self._loaded = False
        # Optional: load model here if it exists.
        # self.model = YOLO(model_path)
    
    def detect(self, frame: np.ndarray) -> bool:
        """
        Runs inference on the frame to find potholes.
        
        Returns:
            bool: True if a pothole is detected in the immediate vehicle path, False otherwise.
        """
        # Placeholder logic: no pothole model is loaded, always return False
        return False

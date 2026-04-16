"""
perception/pothole_detector.py — Detects potholes on the road.

Currently acts as a stub / placeholder so the architecture is fully prepared.
You can replace the contents of detect() with inference from a fine-tuned
YOLOv8 model (e.g. `yolov8_pothole.pt`) when you are ready.
"""

import numpy as np
import os

class PotholeDetector:
    def __init__(self, model_path: str = "pothole_yolov8.pt"):
        self.model_path = model_path
        self._loaded = False
        
        # We only load it if the file exists to avoid crashing if the user hasn't supplied it.
        if os.path.exists(model_path):
            from ultralytics import YOLO
            import torch
            
            print(f"[INFO] Loading Pothole YOLO model from {model_path} ...")
            self.model = YOLO(model_path)
            if torch.cuda.is_available():
                self.model.to('cuda')
            self._loaded = True
        else:
            print(f"[WARN] Pothole model '{model_path}' not found. Pothole detection disabled.")
            self.model = None

    def detect(self, frame: np.ndarray) -> bool:
        """
        Runs inference on the frame to find potholes.
        Returns:
            bool: True if a pothole is detected in the immediate vehicle path.
        """
        if not self._loaded:
            return False
            
        results = self.model(frame, verbose=False)
        for r in results:
            if len(r.boxes) > 0:
                # Basic check: if ANY pothole is detected in the frame, return True.
                # Here you can add logic to only return True if the box is in your path.
                return True
        return False

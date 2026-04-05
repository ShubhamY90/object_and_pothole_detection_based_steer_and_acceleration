"""
perception/detector.py — Object detection via YOLOv8.

Wraps the YOLO model so the rest of the pipeline only deals with clean
Python dataclasses rather than raw ultralytics tensors.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from ultralytics import YOLO
import config


@dataclass
class Detection:
    """Single detected object from one frame."""
    class_id:   int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    # Filled in later by the depth estimator
    depth: float = 0.0

    # Filled in by the tracker
    approaching: bool = False

    @property
    def cx(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def cy(self) -> int:
        return (self.y1 + self.y2) // 2

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def is_human(self) -> bool:
        return self.class_id == 0

    def priority_depth(self) -> float:
        """Depth score used for sorting. Humans get a small boost."""
        boost = config.HUMAN_PRIORITY_BOOST if self.is_human else 0.0
        return self.depth + boost


class ObjectDetector:
    """
    Thin wrapper around YOLOv8.

    Only returns detections for classes listed in config.RELEVANT_CLASSES
    and above config.YOLO_CONF_THRESH.
    """

    def __init__(self, model_path: str = config.YOLO_MODEL):
        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on *frame* (BGR numpy array).
        Returns a list of Detection objects for relevant classes only.
        """
        results = self.model(frame, verbose=False)
        detections: List[Detection] = []

        for r in results:
            for box in r.boxes:
                cls_id  = int(box.cls[0])
                conf    = float(box.conf[0])

                if cls_id not in config.RELEVANT_CLASSES:
                    continue
                if conf < config.YOLO_CONF_THRESH:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(Detection(
                    class_id   = cls_id,
                    class_name = config.RELEVANT_CLASSES[cls_id],
                    confidence = conf,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                ))

        return detections

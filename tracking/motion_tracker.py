"""
tracking/motion_tracker.py — Temporal consistency and approach detection.

Change from original
--------------------
MiDaS relative depth is unreliable for control (Fix 2). We now derive a
"closeness" proxy purely from bounding-box geometry:

    closeness = y_centre / frame_height

Objects lower in the frame (larger y) are geometrically closer. This is
stable, monotonic, and requires zero extra compute.

Each Detection's `.depth` field is set here so the rest of the pipeline
(controller, threat selection) continues to work unchanged — they just get
a better depth estimate.

Extending
---------
* Swap `_bbox_depth` for stereo/LiDAR depth if available.
* Add IoU-based ID tracking to handle the same object shifting positions.
"""

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import numpy as np

import config
from perception.detector import Detection


class MotionTracker:
    """
    Maintains frame-to-frame state needed to decide if an object is
    approaching the vehicle.

    Usage
    -----
    tracker = MotionTracker()
    for frame, detections in stream:
        tracker.update(detections, depth_map=None, frame_height=h)
        # detections now have .approaching and .depth set
    """

    def __init__(self):
        # EMA closeness history keyed by spatial bucket (cx//20, cy//20)
        self._closeness_ema: Dict[Tuple[int, int], float] = {}

        # Short history of the primary threat closeness for spike detection
        self._threat_history: deque = deque(maxlen=5)

        # Last frame's EMA snapshot for delta comparison
        self._prev_ema: Dict[Tuple[int, int], float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections:   List[Detection],
        depth_map:    Optional[np.ndarray] = None,   # kept for API compat, ignored
        frame_height: int                  = config.FRAME_HEIGHT,
    ) -> None:
        """
        Annotate each Detection's `.approaching` and `.depth` in-place.

        Parameters
        ----------
        detections   : list of Detection objects
        depth_map    : ignored (kept for API compatibility)
        frame_height : frame height in pixels, used for closeness normalisation
        """
        for det in detections:
            closeness = self._bbox_depth(det, frame_height)

            # Lift closeness value into .depth so controllers don't change
            det.depth = closeness

            # Spatial bucket for EMA (coarse grid tolerates jitter)
            bucket = (det.cx // 20, det.cy // 20)
            if bucket in self._closeness_ema:
                smoothed = (
                    config.DEPTH_SMOOTH_ALPHA * closeness
                    + (1.0 - config.DEPTH_SMOOTH_ALPHA) * self._closeness_ema[bucket]
                )
            else:
                smoothed = closeness
            self._closeness_ema[bucket] = smoothed

            # Approaching: closeness increased compared to previous frame
            det.approaching = False
            if bucket in self._prev_ema:
                delta = smoothed - self._prev_ema[bucket]
                if delta > config.DEPTH_APPROACHING_THRESHOLD:
                    det.approaching = True

        # Advance state
        self._prev_ema = dict(self._closeness_ema)

    def record_threat(self, depth: float) -> None:
        """
        Call once per frame with the primary threat's depth/closeness
        so the tracker can detect sudden spikes.
        """
        self._threat_history.append(depth)

    def is_spike(self, current_depth: float) -> bool:
        """
        Return True if *current_depth* is anomalously higher than the
        recent rolling average, indicating a noisy reading.

        Spike: current > mean + 3 * std (evaluated after ≥ 3 samples).
        """
        if len(self._threat_history) < 3:
            return False
        arr  = np.array(self._threat_history)
        mean = arr.mean()
        std  = arr.std()
        return bool(current_depth > mean + 3 * std)

    def reset(self) -> None:
        """Clear all state (useful when jumping to a new sequence)."""
        self._closeness_ema.clear()
        self._prev_ema.clear()
        self._threat_history.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_depth(det: Detection, frame_height: int) -> float:
        """
        Geometry-based closeness proxy in [0, 1].

        Two complementary signals blended:
          1. y_centre / frame_height  — objects lower in frame are closer
          2. box_height / frame_height — larger boxes are closer

        Both are scale-free and robust across lighting/weather.
        """
        y_center   = (det.y1 + det.y2) / 2.0
        y_norm     = float(np.clip(y_center   / (frame_height + 1e-6), 0.0, 1.0))
        h_norm     = float(np.clip(det.height / (frame_height + 1e-6), 0.0, 1.0))
        # Weighted blend: y-position dominates (0.6), box size supports (0.4)
        return 0.6 * y_norm + 0.4 * h_norm

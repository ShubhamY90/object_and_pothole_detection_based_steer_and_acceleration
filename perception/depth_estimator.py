"""
perception/depth_estimator.py — Relative depth via MiDaS.

MiDaS outputs *inverse* depth: a higher value means the object is closer.
We normalise the map to [0, 1] so downstream code works with consistent
numbers regardless of the scene's absolute scale.

Adding a new depth backend (e.g. Depth-Anything) is as simple as
subclassing DepthEstimator and overriding `estimate`.
"""

import numpy as np
import torch
import torch.nn.functional as F
import config
from perception.detector import Detection
from typing import List


class DepthEstimator:
    """
    Loads MiDaS and exposes a single `estimate(frame)` method.

    Args:
        model_name: MiDaS variant.  "MiDaS_small" is fast; "DPT_Large"
                    is accurate but slow.
        device:     torch.device to run inference on.
    """

    def __init__(
        self,
        model_name: str = config.MIDAS_MODEL,
        device: torch.device = None,
    ):
        if device is None:
            device = torch.device(
                "mps"  if torch.backends.mps.is_available()  else
                "cuda" if torch.cuda.is_available()          else
                "cpu"
            )
        self.device = device

        self.model = torch.hub.load("intel-isl/MiDaS", model_name)
        self.model.to(self.device)
        self.model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = (
            transforms.small_transform
            if "small" in model_name.lower()
            else transforms.dpt_transform
        )

        # Simple EMA state: previous normalised depth map for smoothing.
        self._prev_map: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Compute a normalised depth map for *frame_bgr*.

        Returns
        -------
        depth_map : np.ndarray, shape (H, W), dtype float32
            Values in [0, 1].  1.0 = closest possible, 0.0 = farthest.
        """
        import cv2
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        batch = self.transform(frame_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(batch)

        # Resize back to original frame dimensions
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=frame_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = prediction.cpu().numpy().astype(np.float32)

        # Normalise to [0, 1]
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        # Optional EMA smoothing to reduce frame-to-frame noise
        if self._prev_map is not None and self._prev_map.shape == depth.shape:
            alpha = config.DEPTH_SMOOTH_ALPHA
            depth = alpha * depth + (1.0 - alpha) * self._prev_map

        self._prev_map = depth.copy()
        return depth

    def annotate_detections(
        self,
        depth_map: np.ndarray,
        detections: List[Detection],
    ) -> None:
        """
        Fill each Detection's `.depth` field in-place using the centre
        pixel of its bounding box.
        """
        h, w = depth_map.shape
        for det in detections:
            cy = np.clip(det.cy, 0, h - 1)
            cx = np.clip(det.cx, 0, w - 1)
            det.depth = float(depth_map[cy, cx])

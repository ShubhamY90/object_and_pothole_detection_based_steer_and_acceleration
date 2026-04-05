"""
perception/road_detector.py — Segmentation-based road/drivable-area detector.

Why replace the classical edge+Hough approach?
-----------------------------------------------
* Canny+Hough detects ANY strong edge — railway tracks, bushes, fences, and
  road markings alike. This causes massive false positives in road masking.
* DeepLabV3 (ResNet-50 backbone, COCO-pretrained) performs semantic
  segmentation and learns WHAT a road surface actually looks like, not just
  edges.  Class 0 in the COCO/VOC label set corresponds to background
  (non-object) which, combined with class 15 (person on a road), gives us a
  robust proxy for the drivable surface.

Output interface (drop-in replacement for LaneResult)
----------------------------------------------------
RoadResult has:
  - road_mask      : binary uint8 (H×W) — white pixels = drivable surface
  - lane_centre_x  : x-centroid of the bottom-half of the road mask (pixels)
  - debug_frame    : annotated BGR frame for visualisation

Model note
----------
DeepLabV3 with MobileNetV3 backbone is used by default for speed. You can
swap to `deeplabv3_resnet101` for better quality at the cost of speed.
The model is auto-downloaded from torchvision on first run (~50 MB).
"""

import cv2
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional

try:
    import torchvision.transforms.functional as TF
    from torchvision.models.segmentation import (
        deeplabv3_mobilenet_v3_large,
        DeepLabV3_MobileNet_V3_Large_Weights,
    )
    _TORCHVISION_OK = True
except ImportError:
    _TORCHVISION_OK = False

import config


# ---------------------------------------------------------------------------
# Data container — mirrors LaneResult so the rest of the pipeline needs no
# changes other than swapping the class.
# ---------------------------------------------------------------------------

@dataclass
class RoadResult:
    road_mask:      Optional[np.ndarray] = None   # uint8 binary H×W  (255 = road)
    lane_centre_x:  Optional[float]     = None    # pixels
    debug_frame:    Optional[np.ndarray] = None

    # Keep these for backward-compatibility with any code that checks them
    left_poly:  object = None
    right_poly: object = None


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class RoadDetector:
    """
    Segmentation-based drivable-area detector.

    Tries to use DeepLabV3 (torchvision).  If torchvision is not installed
    it falls back to a trapezoid heuristic so the pipeline still runs.

    Usage
    -----
    detector = RoadDetector()
    result   = detector.detect(frame_bgr)
    # result.road_mask       → use as drivable area mask in filtering
    # result.lane_centre_x   → steering reference
    """

    # COCO class indices that map to the road surface.
    # DeepLabV3 trained on COCO/VOC: class 0 = background (includes road),
    # class 15 = person. We treat background as road and combine with a
    # bottom-ROI crop to exclude sky/buildings.
    _ROAD_CLASSES = {0}          # extend if needed (e.g. 20 = road in some maps)

    # Inference resolution (width, height). Smaller → faster, coarser mask.
    _INFER_W = 320
    _INFER_H = 320

    def __init__(self):
        self._model  = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if _TORCHVISION_OK:
            print("[RoadDetector] Loading DeepLabV3-MobileNetV3 …")
            weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
            self._model = (
                deeplabv3_mobilenet_v3_large(weights=weights)
                .to(self._device)
                .eval()
            )
            print(f"[RoadDetector] Ready — device={self._device}")
        else:
            print("[RoadDetector] WARNING: torchvision not found — "
                  "using trapezoid fallback. Install with: pip install torchvision")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> RoadResult:
        """
        Run road segmentation on a BGR frame.
        Always returns a RoadResult; falls back to trapezoid if model absent.
        """
        h, w = frame.shape[:2]
        result = RoadResult(debug_frame=frame.copy())

        if self._model is None:
            # Fallback — full trapezoid heuristic
            result.road_mask    = self._trapezoid_mask(w, h)
            result.lane_centre_x = float(w / 2)
            self._apply_roi(result.road_mask, h)
            result.lane_centre_x = self._compute_centre_x(result.road_mask, w, h)
            self._draw_overlay(result.debug_frame, result.road_mask)
            return result

        # ------------------------------------------------------------------
        # 1. Pre-process: BGR → RGB tensor, resize, normalise
        # ------------------------------------------------------------------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to inference resolution
        small = cv2.resize(rgb, (self._INFER_W, self._INFER_H))
        # HWC → CHW, [0,255] → [0.0,1.0]
        tensor = torch.from_numpy(small).permute(2, 0, 1).float() / 255.0
        # ImageNet normalisation expected by DeepLabV3
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.unsqueeze(0).to(self._device)

        # ------------------------------------------------------------------
        # 2. Inference
        # ------------------------------------------------------------------
        with torch.no_grad():
            logits = self._model(tensor)["out"][0]   # shape: (num_classes, H, W)
        pred_cls = logits.argmax(dim=0).cpu().numpy().astype(np.uint8)

        # ------------------------------------------------------------------
        # 3. Build binary road mask
        # ------------------------------------------------------------------
        # "background" class (0) at inference resolution
        road_small = np.zeros_like(pred_cls, dtype=np.uint8)
        for cls_id in self._ROAD_CLASSES:
            road_small[pred_cls == cls_id] = 255

        # Scale back to original frame size
        road_mask = cv2.resize(road_small, (w, h), interpolation=cv2.INTER_NEAREST)

        # ------------------------------------------------------------------
        # 4. ROI — ignore the top 50 % of the frame (sky, far horizon)
        #    This dramatically reduces false positives from buildings/sky.
        # ------------------------------------------------------------------
        self._apply_roi(road_mask, h)

        # ------------------------------------------------------------------
        # 5. Morphological cleanup — close small holes, remove tiny blobs
        # ------------------------------------------------------------------
        kernel = np.ones((7, 7), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN,  kernel)

        # ------------------------------------------------------------------
        # 6. Lane centre estimate from mask centroid in bottom half
        # ------------------------------------------------------------------
        centre_x = self._compute_centre_x(road_mask, w, h)

        result.road_mask     = road_mask
        result.lane_centre_x = centre_x

        self._draw_overlay(result.debug_frame, road_mask)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_roi(mask: np.ndarray, h: int) -> None:
        """Zero-out the top 50 % of the mask in-place.
        Focus only on the near-vehicle surface (Fix 5)."""
        cutoff = int(0.50 * h)
        mask[:cutoff, :] = 0

    @staticmethod
    def _compute_centre_x(mask: np.ndarray, w: int, h: int) -> float:
        """
        X-centroid of road pixels in the bottom third of the frame.
        Falls back to frame centre if no road pixels are found.
        """
        bottom_strip = mask[int(0.67 * h):, :]
        cols = np.where(bottom_strip > 0)[1]
        if cols.size == 0:
            return float(w / 2)
        return float(cols.mean())

    @staticmethod
    def _trapezoid_mask(w: int, h: int) -> np.ndarray:
        """Fallback road mask when segmentation model is absent."""
        poly = np.array([
            [int(config.ROAD_BOT_LEFT_X  * w), h],
            [int(config.ROAD_TOP_LEFT_X  * w), int(config.ROAD_TOP_Y * h)],
            [int(config.ROAD_TOP_RIGHT_X * w), int(config.ROAD_TOP_Y * h)],
            [int(config.ROAD_BOT_RIGHT_X * w), h],
        ], dtype=np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        return mask

    @staticmethod
    def _draw_overlay(frame: np.ndarray, mask: np.ndarray) -> None:
        """Tint road pixels green on the debug frame."""
        overlay = np.zeros_like(frame)
        overlay[mask > 0] = (0, 100, 0)
        cv2.addWeighted(frame, 1.0, overlay, 0.35, 0, dst=frame)
        cv2.putText(
            frame, "SEG-ROAD", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2, cv2.LINE_AA,
        )

"""
filtering/region_filter.py — Lane-aware spatial filtering.

Uses the road mask produced by LaneDetector (or the trapezoid fallback)
to decide whether each detection is on the drivable surface.

Also counts how many on-road obstacles are in each horizontal region so
the controller can steer toward the least-blocked side.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

import config
from perception.detector import Detection


class Region(str, Enum):
    LEFT   = "LEFT"
    CENTER = "CENTER"
    RIGHT  = "RIGHT"


@dataclass
class FilterResult:
    """Everything the controller and renderer need from one filter pass."""
    relevant:        List[Detection]        # on-road threats
    ignored:         List[Detection]        # off-road / above horizon
    counts:          Dict[Region, int]      # obstacle count per region
    least_blocked:   Optional[Region]       # region with fewest obstacles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_region(cx: int, frame_width: int) -> Region:
    rel = cx / frame_width
    if rel < config.LEFT_BOUNDARY:
        return Region.LEFT
    if rel > config.RIGHT_BOUNDARY:
        return Region.RIGHT
    return Region.CENTER


def is_in_near_half(cy: int, frame_height: int) -> bool:
    return cy > config.NEAR_HALF_THRESHOLD * frame_height


def bbox_mask_overlap(det: Detection, mask: np.ndarray) -> float:
    """
    Fraction of the bounding box that falls on white (255) pixels of mask.
    Fast numpy slice — no polygon rasterisation per call.
    """
    h, w = mask.shape
    x1 = max(det.x1, 0);  y1 = max(det.y1, 0)
    x2 = min(det.x2, w);  y2 = min(det.y2, h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi      = mask[y1:y2, x1:x2]
    bbox_px  = roi.size
    road_px  = int(np.count_nonzero(roi))
    return road_px / bbox_px if bbox_px > 0 else 0.0


def _least_blocked_region(counts: Dict[Region, int]) -> Optional[Region]:
    """
    Return the region with fewest on-road obstacles.
    Tie-break: prefer RIGHT or LEFT per config.
    """
    if not counts or all(v == 0 for v in counts.values()):
        return None

    min_count  = min(counts.values())
    candidates = [r for r, c in counts.items() if c == min_count]

    if len(candidates) == 1:
        return candidates[0]

    preferred = Region.RIGHT if config.OBSTACLE_TIEBREAK_PREFER_RIGHT else Region.LEFT
    return preferred if preferred in candidates else candidates[0]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def filter_detections(
    detections:   List[Detection],
    frame_width:  int,
    frame_height: int,
    road_mask:    Optional[np.ndarray] = None,
) -> FilterResult:
    """
    Filter detections using the segmentation-based road mask.

    With the new RoadDetector (DeepLabV3) the mask is semantically accurate:
    railway tracks, bushes, and roadside objects no longer appear as "road".
    The overlap check therefore becomes a strong gating criterion.

    Parameters
    ----------
    detections   : raw detections from ObjectDetector
    frame_width  : pixels
    frame_height : pixels
    road_mask    : uint8 binary (H x W) — 255 = drivable surface.
                   Produced by RoadDetector.detect(); falls back to a
                   trapezoid heuristic if None.

    Returns
    -------
    FilterResult with relevant/ignored lists, per-region counts,
    and the least-blocked region for steering fallback.

    Filter logic (Fix 3)
    --------------------
    A detection is kept ONLY when:
      1. Its bounding-box centre is in the lower 60 % of the frame
         (horizon check — removes sky-level false positives).
      2. At least ROAD_OVERLAP_THRESH fraction of its bounding box
         overlaps white (road) pixels in the mask.
         With a good segmentation mask this reliably removes detections
         on railway tracks, grass, and other off-road surfaces.
    """
    if road_mask is None:
        road_mask = _trapezoid_mask(frame_width, frame_height)

    relevant: List[Detection] = []
    ignored:  List[Detection] = []
    counts:   Dict[Region, int] = {Region.LEFT: 0, Region.CENTER: 0, Region.RIGHT: 0}

    for det in detections:

        # 1. Horizon check
        if not is_in_near_half(det.cy, frame_height):
            det.ignore_reason = "above horizon"
            ignored.append(det)
            continue

        # 2. Road mask overlap check (Fix 3)
        #    Rejects anything not on the segmented drivable surface.
        overlap = bbox_mask_overlap(det, road_mask)
        if overlap < config.ROAD_OVERLAP_THRESH:
            det.ignore_reason = f"off-road ({overlap:.0%})"
            ignored.append(det)
            continue

        # 3. Tag region and count
        det.region = classify_region(det.cx, frame_width)
        counts[det.region] += 1
        relevant.append(det)

    least = _least_blocked_region(counts)
    return FilterResult(
        relevant      = relevant,
        ignored       = ignored,
        counts        = counts,
        least_blocked = least,
    )


def select_primary_threat(detections: List[Detection]) -> "Detection | None":
    """Highest-priority on-road detection (humans boosted)."""
    if not detections:
        return None
    return max(detections, key=lambda d: d.priority_depth())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _trapezoid_mask(w: int, h: int) -> np.ndarray:
    poly = np.array([
        [int(config.ROAD_BOT_LEFT_X  * w), h],
        [int(config.ROAD_TOP_LEFT_X  * w), int(config.ROAD_TOP_Y * h)],
        [int(config.ROAD_TOP_RIGHT_X * w), int(config.ROAD_TOP_Y * h)],
        [int(config.ROAD_BOT_RIGHT_X * w), h],
    ], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask


def get_road_polygon(frame_width: int, frame_height: int) -> np.ndarray:
    """Kept for renderer backward-compatibility."""
    w, h = frame_width, frame_height
    return np.array([
        [int(config.ROAD_TOP_LEFT_X  * w), int(config.ROAD_TOP_Y * h)],
        [int(config.ROAD_TOP_RIGHT_X * w), int(config.ROAD_TOP_Y * h)],
        [int(config.ROAD_BOT_RIGHT_X * w), h],
        [int(config.ROAD_BOT_LEFT_X  * w), h],
    ], dtype=np.int32)
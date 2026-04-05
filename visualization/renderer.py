"""
visualization/renderer.py — All OpenCV drawing in one place.

Note on lane_result type
------------------------
Functions accept any object that has the attributes:
  .road_mask      (np.ndarray or None)
  .debug_frame    (np.ndarray or None)
  .lane_centre_x  (float or None)

Both LaneResult (classic) and RoadResult (segmentation) satisfy this
interface, so no code changes are needed here beyond the import.
"""

from typing import Dict, List, Optional, Union
import cv2
import numpy as np

import config
from control.controller import VehicleState
from filtering.region_filter import Region, FilterResult
from perception.detector import Detection
from perception.lane_detector import LaneResult
from perception.road_detector import RoadResult

# Convenience alias used in type hints below
_AnyRoadResult = Union[LaneResult, RoadResult]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _label_color(det: Detection) -> tuple:
    if det.approaching:
        return config.COLOR_APPROACH
    return config.COLOR_PERSON if det.is_human else config.COLOR_VEHICLE


def draw_lane_overlay(frame: np.ndarray, lane_result: _AnyRoadResult) -> np.ndarray:
    """
    Draw:
      - semi-transparent green road mask
      - yellow fitted lane lines
      - cyan centre-line marker
    """
    if lane_result is None:
        return frame

    h, w = frame.shape[:2]

    # Road mask fill
    if lane_result.road_mask is not None:
        overlay = frame.copy()
        green   = np.zeros_like(frame)
        green[lane_result.road_mask > 0] = (0, 50, 0)
        cv2.addWeighted(green, 0.35, frame, 0.65, 0, frame)

        # Mask outline
        contours, _ = cv2.findContours(
            lane_result.road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(frame, contours, -1, (0, 200, 80), 2)

    # Lane lines (copied from debug_frame if available)
    if lane_result.debug_frame is not None:
        # Blend the lane lines drawn by LaneDetector._draw_lanes
        # We do this by extracting just the non-black pixels from debug_frame
        diff_mask = cv2.absdiff(lane_result.debug_frame, frame)
        gray_diff = cv2.cvtColor(diff_mask, cv2.COLOR_BGR2GRAY)
        _, mask   = cv2.threshold(gray_diff, 10, 255, cv2.THRESH_BINARY)
        frame[mask > 0] = lane_result.debug_frame[mask > 0]

    # Lane centre marker
    if lane_result.lane_centre_x is not None:
        cx = int(lane_result.lane_centre_x)
        cv2.line(frame, (cx, h), (cx, int(h * 0.7)), (255, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "LANE CTR", (cx + 5, int(h * 0.75)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)

    return frame


def draw_region_lines(frame: np.ndarray) -> np.ndarray:
    """LEFT / CENTER / RIGHT dividers."""
    h, w  = frame.shape[:2]
    left_x  = int(w * config.LEFT_BOUNDARY)
    right_x = int(w * config.RIGHT_BOUNDARY)
    color   = config.COLOR_REGION

    cv2.line(frame, (left_x,  0), (left_x,  h), color, 1, cv2.LINE_AA)
    cv2.line(frame, (right_x, 0), (right_x, h), color, 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "LEFT",   (10,          20), font, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, "CENTER", (left_x + 5,  20), font, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(frame, "RIGHT",  (right_x + 5, 20), font, 0.5, color, 1, cv2.LINE_AA)
    return frame


def draw_obstacle_counts(
    frame:  np.ndarray,
    counts: Dict[Region, int],
    least_blocked: Optional[Region],
) -> np.ndarray:
    """Show per-region obstacle counts at the top of each zone."""
    h, w  = frame.shape[:2]
    left_x  = int(w * config.LEFT_BOUNDARY)
    right_x = int(w * config.RIGHT_BOUNDARY)
    font    = cv2.FONT_HERSHEY_SIMPLEX

    positions = {
        Region.LEFT:   (left_x  // 2,            40),
        Region.CENTER: (left_x + (right_x - left_x) // 2, 40),
        Region.RIGHT:  (right_x + (w - right_x) // 2,     40),
    }

    for region, pos in positions.items():
        count = counts.get(region, 0)
        is_least = (region == least_blocked)
        color = (0, 255, 0) if is_least else (200, 200, 200)
        suffix = " ←STEER" if is_least else ""
        cv2.putText(frame, f"{count} obj{suffix}", pos, font, 0.5, color, 1, cv2.LINE_AA)

    return frame


def draw_detections(
    frame:      np.ndarray,
    detections: List[Detection],
    ignored:    List[Detection],
) -> np.ndarray:
    """Coloured boxes for on-road threats, dim boxes for ignored."""
    for det in ignored:
        reason = getattr(det, "ignore_reason", "ignored")
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), (55, 55, 55), 1)
        cv2.putText(frame, reason, (det.x1, det.y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)

    for det in detections:
        color  = _label_color(det)
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)

        region_str = getattr(det, "region", Region.CENTER)
        if hasattr(region_str, "value"):
            region_str = region_str.value
        approach = " ⚠" if det.approaching else ""
        label = f"{det.class_name} d={det.depth:.2f} [{region_str}]{approach}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        tx, ty = det.x1, max(det.y1 - 4, th + 4)
        cv2.rectangle(frame, (tx, ty - th - 4), (tx + tw + 4, ty + 2), color, cv2.FILLED)
        cv2.putText(frame, label, (tx + 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


def draw_hud(
    frame:  np.ndarray,
    state:  VehicleState,
    threat: Optional[Detection],
) -> np.ndarray:
    """Bottom-left HUD panel."""
    h, w  = frame.shape[:2]
    font  = cv2.FONT_HERSHEY_DUPLEX

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 170), (340, h), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    base_y = h - 170 + 28
    gap    = 28
    white  = config.COLOR_HUD_TEXT
    warn   = config.COLOR_HUD_WARN

    cv2.putText(frame, f"SPEED : {state.velocity:5.1f} km/h",   (10, base_y),           font, 0.60, white, 1, cv2.LINE_AA)
    cv2.putText(frame, f"ACCEL : {state.acceleration:+6.1f}",   (10, base_y + gap),      font, 0.60, white, 1, cv2.LINE_AA)
    cv2.putText(frame, f"ACTION: {state.action_label}",         (10, base_y + 2 * gap),  font, 0.60,
                warn if "BRAKE" in state.action_label else white, 1, cv2.LINE_AA)
    cv2.putText(frame, f"STEER : {state.steering:+.2f}  {_steer_label(state.steering)}", (10, base_y + 3 * gap), font, 0.60, white, 1, cv2.LINE_AA)
    cv2.putText(frame, f"REASON: {state.steer_reason}",         (10, base_y + 4 * gap),  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    if threat and threat.approaching:
        cv2.putText(frame, "▲ APPROACHING", (10, base_y + 5 * gap),
                    font, 0.70, warn, 2, cv2.LINE_AA)

    return frame


def render(
    frame:         np.ndarray,
    filter_result: FilterResult,
    state:         VehicleState,
    threat:        Optional[Detection],
    lane_result:   Optional[_AnyRoadResult] = None,
    show_regions:  bool = True,
) -> np.ndarray:
    """Master render call."""
    annotated = frame.copy()

    # 1. Lane overlay (road mask + fitted lines)
    if lane_result is not None:
        draw_lane_overlay(annotated, lane_result)

    # 2. Region dividers
    if show_regions:
        draw_region_lines(annotated)

    # 3. Obstacle counts per region
    draw_obstacle_counts(annotated, filter_result.counts, filter_result.least_blocked)

    # 4. Bounding boxes
    draw_detections(annotated, filter_result.relevant, filter_result.ignored)

    # 5. HUD
    draw_hud(annotated, state, threat)

    return annotated


def _steer_label(steer: float) -> str:
    if steer > 0.15:  return "→ RIGHT"
    if steer < -0.15: return "← LEFT"
    return "↑ STRAIGHT"

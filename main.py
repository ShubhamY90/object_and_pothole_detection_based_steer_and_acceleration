"""
main.py — Autonomous driving pipeline entry point.

Pipeline per frame:
    1. PERCEPTION   — YOLO detection + Road segmentation (DeepLabV3)
    2. FILTERING    — segmentation road-mask overlap + per-region counts
    3. TRACKING     — approaching detection via bbox-geometry depth
    4. CONTROL      — safe-centre steering + smoothed speed/steer
    5. VISUALISE    — road overlay + HUD + bounding boxes

Fixes applied
-------------
    Fix 1 — Replaced edge-based LaneDetector with DeepLabV3 RoadDetector.
    Fix 2 — Removed MiDaS; tracker computes bbox-geometry depth.
    Fix 3 — Road-mask filtering now benefits from semantic segmentation.
    Fix 4 — Controller uses safe-centre logic for obstacle avoidance.
    Fix 5 — ROI applied inside RoadDetector (top 50 % zeroed).
    Fix 6 — EMA smoothing on steer and speed.
    Fix 7 — Speed reduced proportionally to |steer|.

Usage
-----
    python main.py --data data/kitti/image_02/data
    python main.py --data data/kitti/image_02/data --save output.mp4
    python main.py --data data/kitti/image_02/data --no-display
"""

import argparse
import sys
import os
import cv2

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

import config
from filtering     import filter_detections, select_primary_threat, FilterResult
from tracking      import MotionTracker
from decision      import BehaviorPlanner
from perception.pothole_detector import PotholeDetector
from control       import VehicleController
from visualization import render
from utils         import iter_frames


# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------

def build_pipeline():
    print("[INFO] Loading YOLOv8 …")
    detector = ObjectDetector(config.YOLO_MODEL)

    # Fix 1 — Use segmentation-based RoadDetector instead of LaneDetector
    print("[INFO] Loading DeepLabV3 road detector …")
    road_detector = RoadDetector()

    # Fix 2 — MotionTracker now uses bbox-geometry depth; no MiDaS needed
    tracker    = MotionTracker()
    print("[INFO] Loading Pothole detector …")
    pothole_detector = PotholeDetector()
    planner    = BehaviorPlanner()
    controller = VehicleController(dt=config.DT)

    return detector, road_detector, pothole_detector, tracker, planner, controller


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    data_folder:  str,
    display:      bool = True,
    save_path:    str  = "",
    window_title: str  = "Autonomous Driving Pipeline",
):
    detector, road_detector, tracker, controller = build_pipeline()

    if save_path:
        output_dir = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] Output directory ready: {output_dir}")

    writer = None

    for filename, frame in iter_frames(data_folder):
        h, w = frame.shape[:2]

        # ------------------------------------------------------------------
        # 1. PERCEPTION
        #    Fix 1: RoadDetector returns a RoadResult with `road_mask` and
        #    `lane_centre_x` — same interface as old LaneResult.
        # ------------------------------------------------------------------
        detections  = detector.detect(frame)
        road_result = road_detector.detect(frame)

        # ------------------------------------------------------------------
        # 2. FILTERING  (semantic road mask from RoadDetector)
        #    Fix 3: segmentation mask removes railway tracks, bushes, etc.
        # ------------------------------------------------------------------
        filter_result = filter_detections(
            detections   = detections,
            frame_width  = w,
            frame_height = h,
            road_mask    = road_result.road_mask,
        )

        # ------------------------------------------------------------------
        # 3. TRACKING
        #    Fix 2: depth_map=None — tracker derives depth from bbox geometry.
        # ------------------------------------------------------------------
        tracker.update(filter_result.relevant, depth_map=None, frame_height=h)

        # ------------------------------------------------------------------
        # 4. THREAT SELECTION
        # ------------------------------------------------------------------
        threat = select_primary_threat(filter_result.relevant)

        spike = False
        if threat is not None:
            tracker.record_threat(threat.depth)
            spike = tracker.is_spike(threat.depth)

        # ------------------------------------------------------------------
        # 5.5 Pothole Detection & Decision Layer
        # ------------------------------------------------------------------
        pothole_detected = pothole_detector.detect(frame)
        
        directive = planner.plan(
            threat=threat,
            filter_result=filter_result,
            lane_result=road_result,
            pothole_detected=pothole_detected,
            frame_width=w,
            frame_height=h,
            is_spike=spike,
        )

        # ------------------------------------------------------------------
        # 6. Control (Execution)
        # ------------------------------------------------------------------
        state = controller.step(directive)

        # ------------------------------------------------------------------
        # 6. VISUALISATION
        #    road_result is passed as lane_result — renderer uses .road_mask
        #    and .debug_frame, both present on RoadResult.
        # ------------------------------------------------------------------
        annotated = render(
            frame         = frame,
            filter_result = filter_result,
            state         = state,
            threat        = threat,
            lane_result   = road_result,    # duck-typed — same attributes used
            show_regions  = True,
        )

        print(f"[{filename}]  {state}")

        # ------------------------------------------------------------------
        # 7. DISPLAY / SAVE
        # ------------------------------------------------------------------
        if display:
            cv2.imshow(window_title, annotated)
            key = cv2.waitKey(30)
            if key == 27:
                break
            if key == ord("p"):
                cv2.waitKey(0)

        if save_path:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(save_path, fourcc, 10, (w, h))
            writer.write(annotated)

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Modular autonomous driving prototype"
    )
    parser.add_argument("--data", "-d", default="data/kitti/image_02/data")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--save", "-s", default="", metavar="OUTPUT.mp4")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isdir(args.data):
        print(f"[ERROR] Data folder not found: '{args.data}'")
        print("        Pass correct path with --data")
        sys.exit(1)

    run(
        data_folder = args.data,
        display     = not args.no_display,
        save_path   = args.save,
    )

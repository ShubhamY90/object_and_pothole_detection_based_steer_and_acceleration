"""
main2.py — Autonomous driving pipeline entry point (ESP32 camera stream).

Identical pipeline to main.py; the ONLY change is the input source:
instead of iterating over a folder of images we read a live MJPEG stream
from an ESP32-CAM at the URL configured below.

Pipeline per frame:
    1. PERCEPTION   — YOLO detection + Road segmentation (DeepLabV3)
    2. FILTERING    — segmentation road-mask overlap + per-region counts
    3. TRACKING     — approaching detection via bbox-geometry depth
    4. CONTROL      — safe-centre steering + smoothed speed/steer
    5. VISUALISE    — road overlay + HUD + bounding boxes

Fixes applied (inherited from main.py)
--------------------------------------
    Fix 1 — Replaced edge-based LaneDetector with DeepLabV3 RoadDetector.
    Fix 2 — Removed MiDaS; tracker computes bbox-geometry depth.
    Fix 3 — Road-mask filtering now benefits from semantic segmentation.
    Fix 4 — Controller uses safe-centre logic for obstacle avoidance.
    Fix 5 — ROI applied inside RoadDetector (top 50 % zeroed).
    Fix 6 — EMA smoothing on steer and speed.
    Fix 7 — Speed reduced proportionally to |steer|.

Usage
-----
    python main2.py
    python main2.py --stream http://espcam.local:81/stream
    python main2.py --stream http://espcam.local:81/stream --save output.mp4
    python main2.py --stream http://espcam.local:81/stream --no-display
"""

import argparse
import sys
import os

import cv2
import requests
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

import config
from perception    import ObjectDetector, RoadDetector
from filtering     import filter_detections, select_primary_threat, FilterResult
from tracking      import MotionTracker
from control       import VehicleController
from visualization import render
from led_signal    import send_leds


# ---------------------------------------------------------------------------
# Pipeline setup  (identical to main.py)
# ---------------------------------------------------------------------------

def build_pipeline():
    print("[INFO] Loading YOLOv8 …")
    detector = ObjectDetector(config.YOLO_MODEL)

    print("[INFO] Loading DeepLabV3 road detector …")
    road_detector = RoadDetector()

    tracker    = MotionTracker()
    controller = VehicleController(dt=config.DT)

    return detector, road_detector, tracker, controller


# ---------------------------------------------------------------------------
# Main loop — ESP32 MJPEG stream input
# ---------------------------------------------------------------------------

def run(
    stream_url:   str  = "http://espcam.local:81/stream",
    display:      bool = True,
    save_path:    str  = "",
    window_title: str  = "Autonomous Driving Pipeline — ESP32 Stream",
):
    detector, road_detector, tracker, controller = build_pipeline()

    if save_path:
        output_dir = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] Output directory ready: {output_dir}")

    writer = None

    # -----------------------------------------------------------------------
    # Open the MJPEG stream
    # -----------------------------------------------------------------------
    print(f"[INFO] Connecting to stream: {stream_url}")
    try:
        stream = requests.get(stream_url, stream=True, timeout=10)
        stream.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"[ERROR] Could not connect to stream: {exc}")
        sys.exit(1)
    print("[INFO] Stream connected. Press ESC to quit.")

    bytes_data = b""
    frame_idx  = 0

    for chunk in stream.iter_content(chunk_size=1024):
        bytes_data += chunk

        # Fix 3 — prevent buffer from growing unbounded on corrupt/stalled streams
        if len(bytes_data) > 100_000:
            bytes_data = b""
            continue

        # Fix 4 — search for end-marker AFTER start-marker to avoid cross-frame matches
        a = bytes_data.find(b"\xff\xd8")           # JPEG SOI marker
        b = bytes_data.find(b"\xff\xd9", a)         # JPEG EOI marker (search after a)

        if a == -1 or b == -1:
            continue                                # incomplete frame — keep buffering

        try:
            jpg        = bytes_data[a : b + 2]
            bytes_data = bytes_data[b + 2 :]       # discard consumed bytes

            # Fix 1 — guard against empty slice
            if len(jpg) == 0:
                continue

            np_arr = np.frombuffer(jpg, dtype=np.uint8)

            # Fix 1 — guard against zero-size array
            if np_arr.size == 0:
                continue

            # Fix 2 — imdecode wrapped in try/except (see outer except)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # ---------------------------------------------------------------
            # Preprocessing — resize to the resolution the pipeline expects.
            # config.FRAME_WIDTH / FRAME_HEIGHT = 640 × 480 (matches main.py).
            # ---------------------------------------------------------------
            frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            h, w  = frame.shape[:2]

            frame_idx += 1
            label = f"frame_{frame_idx:06d}"

            # -----------------------------------------------------------
            # 1. PERCEPTION
            # -----------------------------------------------------------
            detections  = detector.detect(frame)
            road_result = road_detector.detect(frame)

            # -----------------------------------------------------------
            # 2. FILTERING
            # -----------------------------------------------------------
            filter_result = filter_detections(
                detections   = detections,
                frame_width  = w,
                frame_height = h,
                road_mask    = road_result.road_mask,
            )

            # -----------------------------------------------------------
            # 3. TRACKING
            # -----------------------------------------------------------
            tracker.update(filter_result.relevant, depth_map=None, frame_height=h)

            # -----------------------------------------------------------
            # 4. THREAT SELECTION
            # -----------------------------------------------------------
            threat = select_primary_threat(filter_result.relevant)

            spike = False
            if threat is not None:
                tracker.record_threat(threat.depth)
                spike = tracker.is_spike(threat.depth)

            # -----------------------------------------------------------
            # 5. CONTROL
            # -----------------------------------------------------------
            state = controller.step(
                threat        = threat,
                filter_result = filter_result,
                lane_result   = road_result,
                frame_width   = w,
                frame_height  = h,
                is_spike      = spike,
            )

            # -----------------------------------------------------------
            # ★ EXTENSION POINT ★
            # pothole_mask = pothole_detector.detect(frame)
            # state = controller.apply_pothole_override(state, pothole_mask)
            # -----------------------------------------------------------

            # -----------------------------------------------------------
            # 6. VISUALISATION
            # -----------------------------------------------------------
            annotated = render(
                frame         = frame,
                filter_result = filter_result,
                state         = state,
                threat        = threat,
                lane_result   = road_result,
                show_regions  = True,
            )

            print(f"[{label}]  {state}")
            send_leds(state.velocity, state.steering)

            # -----------------------------------------------------------
            # 7. DISPLAY / SAVE
            # -----------------------------------------------------------
            if display:
                cv2.imshow(window_title, annotated)
                key = cv2.waitKey(1)      # 1 ms — keep up with live stream
                if key == 27:             # ESC → quit
                    break
                if key == ord("p"):
                    cv2.waitKey(0)        # pause until any key

            if save_path:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(save_path, fourcc, 10, (w, h))
                writer.write(annotated)

        except Exception:
            # Fix 2 — swallow any decode / pipeline error on a corrupt frame
            continue

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Autonomous driving pipeline — ESP32-CAM MJPEG stream input"
    )
    parser.add_argument(
        "--stream", "-s",
        default="http://espcam.local:81/stream",
        metavar="URL",
        help="MJPEG stream URL from ESP32-CAM (default: %(default)s)",
    )
    parser.add_argument("--no-display", action="store_true",
                        help="Suppress the live preview window")
    parser.add_argument("--save", default="", metavar="OUTPUT.mp4",
                        help="Optional path to save the annotated video")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run(
        stream_url = args.stream,
        display    = not args.no_display,
        save_path  = args.save,
    )

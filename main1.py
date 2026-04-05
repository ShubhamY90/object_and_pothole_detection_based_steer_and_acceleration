"""
main1.py — Fast-SCNN autonomous driving pipeline + FPS benchmark.

Differences from main.py
-------------------------
* RoadDetector (DeepLabV3-MobileNet)  →  FastSCNNRoadDetector
* Every frame is timed precisely; FPS is displayed on the HUD in real-time.
* A rolling benchmark summary is printed to the terminal every 30 frames.
* All other pipeline stages (YOLO, filtering, tracking, control, render)
  are identical to main.py so the comparison is apples-to-apples.

FPS overlay
-----------
Upper-right panel shows:
    SEG FPS  : Fast-SCNN segmentation only
    PIPE FPS : Full pipeline FPS (detection + seg + track + control + render)
    YOLO ms  : YOLO detection latency
    SEG  ms  : Fast-SCNN latency
    TOTAL ms : End-to-end frame latency

Usage
-----
    python main1.py --data data/kitti/image_02/data
    python main1.py --data data/kitti/image_02/data --save output_fastscnn.mp4
    python main1.py --data data/kitti/image_02/data --no-display
    python main1.py --data data/kitti/image_02/data --infer-size 256x128
"""

import argparse
import sys
import os
import time
from collections import deque

import cv2
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

import config
from perception                         import ObjectDetector
from perception.road_detector_fastscnn  import FastSCNNRoadDetector
from filtering                          import filter_detections, select_primary_threat
from tracking                           import MotionTracker
from control                            import VehicleController
from visualization                      import render
from utils                              import iter_frames


# ---------------------------------------------------------------------------
# Rolling-window FPS tracker
# ---------------------------------------------------------------------------

class FPSTracker:
    """Computes rolling-window mean and current FPS / latency."""

    def __init__(self, window: int = 30):
        self._times:    deque = deque(maxlen=window)
        self._seg_ms:   deque = deque(maxlen=window)
        self._yolo_ms:  deque = deque(maxlen=window)
        self._total_ms: deque = deque(maxlen=window)
        self._t0: float = 0.0
        self.last_mask_source: str = "?"

    def frame_start(self) -> None:
        self._t0 = time.perf_counter()

    def record(
        self,
        yolo_ms:     float,
        seg_ms:      float,
        total_ms:    float,
        mask_source: str = "?",
    ) -> None:
        self._yolo_ms.append(yolo_ms)
        self._seg_ms.append(seg_ms)
        self._total_ms.append(total_ms)
        self.last_mask_source = mask_source

    # ── rolling averages ────────────────────────────────────────────────

    @property
    def seg_fps(self) -> float:
        if not self._seg_ms:
            return 0.0
        return 1000.0 / (sum(self._seg_ms) / len(self._seg_ms) + 1e-9)

    @property
    def pipe_fps(self) -> float:
        if not self._total_ms:
            return 0.0
        return 1000.0 / (sum(self._total_ms) / len(self._total_ms) + 1e-9)

    @property
    def mean_yolo_ms(self) -> float:
        return sum(self._yolo_ms) / len(self._yolo_ms) if self._yolo_ms else 0.0

    @property
    def mean_seg_ms(self) -> float:
        return sum(self._seg_ms) / len(self._seg_ms) if self._seg_ms else 0.0

    @property
    def mean_total_ms(self) -> float:
        return sum(self._total_ms) / len(self._total_ms) if self._total_ms else 0.0

    @property
    def frame_count(self) -> int:
        return len(self._total_ms)


# ---------------------------------------------------------------------------
# FPS overlay drawing
# ---------------------------------------------------------------------------

def _draw_fps_panel(
    frame:        np.ndarray,
    tracker:      FPSTracker,
    cur_seg_ms:   float,
    cur_total_ms: float,
) -> None:
    """
    Draw a semi-transparent FPS/latency panel in the top-right corner.
    Shows both the current frame value and the rolling-window average.
    """
    h, w = frame.shape[:2]
    pw, ph = 290, 178           # panel dimensions (extra row for mask source)
    x0, y0 = w - pw - 8, 8     # top-right position

    # Dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), (10, 10, 10), cv2.FILLED)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # Border
    cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), (0, 200, 80), 1)

    font   = cv2.FONT_HERSHEY_SIMPLEX
    small  = cv2.FONT_HERSHEY_PLAIN
    green  = (0, 255, 80)
    yellow = (0, 230, 255)
    white  = (220, 220, 220)
    lh     = 24   # line height

    lines = [
        ("FAST-SCNN PIPELINE",   green,  0.50, 1),
        (f"SEG  FPS : {tracker.seg_fps:6.1f}",  yellow, 0.52, 1),
        (f"PIPE FPS : {tracker.pipe_fps:6.1f}", green,  0.52, 1),
        (f"YOLO  ms : {tracker.mean_yolo_ms:6.1f}", white, 0.48, 1),
        (f"SEG   ms : {cur_seg_ms:6.1f}  (avg {tracker.mean_seg_ms:.1f})", white, 0.45, 1),
        (f"TOTAL ms : {cur_total_ms:6.1f}  (avg {tracker.mean_total_ms:.1f})", white, 0.45, 1),
        (f"MASK SRC : {tracker.last_mask_source}", white, 0.45, 1),
    ]

    for i, (txt, col, scale, thick) in enumerate(lines):
        ty = y0 + 18 + i * lh
        cv2.putText(frame, txt, (x0 + 8, ty), font, scale, col, thick, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Pipeline setup
# ---------------------------------------------------------------------------

def build_pipeline(infer_size: tuple[int, int] | None = None):
    print("[INFO] Loading YOLOv8 …")
    detector = ObjectDetector(config.YOLO_MODEL)

    print("[INFO] Loading Fast-SCNN road detector …")
    road_detector = FastSCNNRoadDetector()

    # Override inference resolution if requested
    if infer_size is not None:
        road_detector._INFER_W, road_detector._INFER_H = infer_size
        print(f"[INFO] Fast-SCNN inference resolution overridden to {infer_size}")

    tracker    = MotionTracker()
    controller = VehicleController(dt=config.DT)
    fps_tracker = FPSTracker(window=30)

    return detector, road_detector, tracker, controller, fps_tracker


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    data_folder:  str,
    display:      bool          = True,
    save_path:    str           = "",
    window_title: str           = "Fast-SCNN Pipeline — FPS Benchmark",
    infer_size:   tuple | None  = None,
):
    detector, road_detector, tracker, controller, fps_tracker = build_pipeline(infer_size)

    if save_path:
        output_dir = os.path.dirname(os.path.abspath(save_path))
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] Output directory ready: {output_dir}")

    writer      = None
    frame_idx   = 0

    for filename, frame in iter_frames(data_folder):
        t_frame_start = time.perf_counter()
        h, w = frame.shape[:2]

        # ------------------------------------------------------------------
        # 1. YOLO Detection
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        detections = detector.detect(frame)
        yolo_ms = (time.perf_counter() - t0) * 1000.0

        # ------------------------------------------------------------------
        # 2. Fast-SCNN Road Segmentation
        # ------------------------------------------------------------------
        t0 = time.perf_counter()
        road_result = road_detector.detect(frame)
        seg_ms = (time.perf_counter() - t0) * 1000.0

        # ------------------------------------------------------------------
        # 3. Filtering
        # ------------------------------------------------------------------
        filter_result = filter_detections(
            detections   = detections,
            frame_width  = w,
            frame_height = h,
            road_mask    = road_result.road_mask,
        )

        # ------------------------------------------------------------------
        # 4. Tracking
        # ------------------------------------------------------------------
        tracker.update(filter_result.relevant, depth_map=None, frame_height=h)

        # ------------------------------------------------------------------
        # 5. Threat selection
        # ------------------------------------------------------------------
        threat = select_primary_threat(filter_result.relevant)
        spike  = False
        if threat is not None:
            tracker.record_threat(threat.depth)
            spike = tracker.is_spike(threat.depth)

        # ------------------------------------------------------------------
        # 6. Control
        # ------------------------------------------------------------------
        state = controller.step(
            threat        = threat,
            filter_result = filter_result,
            lane_result   = road_result,
            frame_width   = w,
            frame_height  = h,
            is_spike      = spike,
        )

        # ------------------------------------------------------------------
        # 7. Visualisation
        # ------------------------------------------------------------------
        annotated = render(
            frame         = frame,
            filter_result = filter_result,
            state         = state,
            threat        = threat,
            lane_result   = road_result,
            show_regions  = True,
        )

        # ------------------------------------------------------------------
        # 8. FPS Overlay
        # ------------------------------------------------------------------
        total_ms = (time.perf_counter() - t_frame_start) * 1000.0
        fps_tracker.record(
            yolo_ms=yolo_ms, seg_ms=seg_ms, total_ms=total_ms,
            mask_source=road_result.mask_source,
        )
        _draw_fps_panel(annotated, fps_tracker, seg_ms, total_ms)

        # ------------------------------------------------------------------
        # 9. Terminal benchmark report every 30 frames
        # ------------------------------------------------------------------
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(
                f"[BENCH #{frame_idx:04d}]  "
                f"PIPE={fps_tracker.pipe_fps:5.1f} FPS  "
                f"SEG={fps_tracker.seg_fps:5.1f} FPS  "
                f"YOLO={fps_tracker.mean_yolo_ms:5.1f}ms  "
                f"SEG={fps_tracker.mean_seg_ms:5.1f}ms  "
                f"TOTAL={fps_tracker.mean_total_ms:5.1f}ms  "
                f"MASK={fps_tracker.last_mask_source}"
            )
        else:
            print(
                f"[{filename}]  {state}  "
                f"| seg={seg_ms:.1f}ms  "
                f"mask={road_result.mask_source}  "
                f"pipe={fps_tracker.pipe_fps:.1f}fps"
            )

        # ------------------------------------------------------------------
        # 10. Display / Save
        # ------------------------------------------------------------------
        if display:
            cv2.imshow(window_title, annotated)
            key = cv2.waitKey(1)   # minimal wait for max FPS
            if key == 27:
                break
            if key == ord("p"):
                cv2.waitKey(0)

        if save_path:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(save_path, fourcc, 10, (w, h))
            writer.write(annotated)

    # Final summary
    print("\n" + "=" * 60)
    print("  FAST-SCNN BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Frames processed : {frame_idx}")
    print(f"  Inference res    : {road_detector._INFER_W}×{road_detector._INFER_H}")
    print(f"  Device           : {road_detector._device}")
    print(f"  Weights loaded   : {road_detector._weights_loaded}")
    print(f"  Last mask source : {fps_tracker.last_mask_source}")
    print(f"  YOLO avg latency : {fps_tracker.mean_yolo_ms:.1f} ms")
    print(f"  SEG  avg latency : {fps_tracker.mean_seg_ms:.1f} ms")
    print(f"  Total avg latency: {fps_tracker.mean_total_ms:.1f} ms")
    print(f"  Seg-only FPS     : {fps_tracker.seg_fps:.1f}")
    print(f"  Full pipeline FPS: {fps_tracker.pipe_fps:.1f}")
    print("=" * 60)

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fast-SCNN autonomous driving pipeline + FPS benchmark"
    )
    parser.add_argument("--data", "-d", default="data/kitti/image_02/data",
                        help="Path to folder of input frames")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable OpenCV window (headless / max FPS test)")
    parser.add_argument("--save", "-s", default="", metavar="OUTPUT.mp4",
                        help="Save annotated output to this video file")
    parser.add_argument(
        "--infer-size", default=None, metavar="WxH",
        help=(
            "Fast-SCNN inference resolution, e.g. 512x256 (default) "
            "or 256x128 (fastest). Format: WIDTHxHEIGHT"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isdir(args.data):
        print(f"[ERROR] Data folder not found: '{args.data}'")
        print("        Pass correct path with --data")
        sys.exit(1)

    infer_size = None
    if args.infer_size:
        try:
            iw, ih = map(int, args.infer_size.lower().split("x"))
            infer_size = (iw, ih)
        except ValueError:
            print(f"[ERROR] --infer-size must be WxH, e.g. 512x256. Got: {args.infer_size}")
            sys.exit(1)

    run(
        data_folder = args.data,
        display     = not args.no_display,
        save_path   = args.save,
        infer_size  = infer_size,
    )

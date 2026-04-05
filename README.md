# Autonomous Driving Prototype

A clean, modular Python pipeline for prototype autonomous driving using
YOLOv8 + MiDaS on the KITTI dataset.

---

## Architecture

```
main.py
│
├── utils/frame_loader.py       I/O: KITTI images or webcam stream
│
├── perception/
│   ├── detector.py             YOLOv8 object detection → Detection dataclass
│   └── depth_estimator.py      MiDaS relative depth, EMA smoothing
│
├── filtering/
│   └── region_filter.py        Class filter + LEFT/CENTER/RIGHT spatial zones
│
├── tracking/
│   └── motion_tracker.py       Frame-to-frame depth delta → "approaching" flag
│                               + spike detection
│
├── control/
│   └── controller.py           Acceleration (4-regime) + steering + physics sim
│
├── visualization/
│   └── renderer.py             Bounding boxes, region lines, HUD overlay
│
└── config.py                   ALL tunable constants in one place
```

### Data flow per frame

```
frame
  │
  ▼
ObjectDetector.detect()          → List[Detection]  (filtered by class + conf)
  │
  ▼
DepthEstimator.estimate()        → depth_map (H×W float32, normalised 0-1)
DepthEstimator.annotate()        → fills Detection.depth in-place
  │
  ▼
filter_detections()              → relevant[], ignored[]
                                   adds Detection.region (LEFT/CENTER/RIGHT)
  │
  ▼
MotionTracker.update()           → fills Detection.approaching in-place
  │
  ▼
select_primary_threat()          → single Detection (highest priority depth)
  │
  ▼
VehicleController.step()         → VehicleState (velocity, acceleration, steer)
  │
  ▼
render()                         → annotated BGR frame
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run on KITTI sequence
python main.py --data /path/to/kitti/image_02/data

# 3. Save output video
python main.py --data /path/to/kitti/image_02/data --save output.mp4

# 4. Headless (server / CI)
python main.py --data /path/to/kitti/image_02/data --no-display
```

### Keyboard shortcuts (during playback)
| Key | Action |
|-----|--------|
| ESC | Quit   |
| P   | Pause / unpause |

---

## Configuration

All constants live in `config.py`.  Key knobs:

| Constant | Default | Effect |
|----------|---------|--------|
| `YOLO_MODEL` | `yolov8n.pt` | Swap to `yolov8s.pt` for better accuracy |
| `MIDAS_MODEL` | `MiDaS_small` | Swap to `DPT_Large` for better depth |
| `DEPTH_HARD_BRAKE` | 0.70 | Depth threshold for emergency brake |
| `NEAR_HALF_THRESHOLD` | 0.40 | Ignore objects in top N% of frame |
| `HUMAN_PRIORITY_BOOST` | 0.10 | Extra depth score added for pedestrians |
| `DEPTH_APPROACHING_THRESHOLD` | 0.025 | Minimum delta to flag as approaching |

---

## Extending the pipeline

### Add pothole detection
1. Create `perception/pothole_detector.py` with a `PotholeDetector` class.
2. Call it in `main.py` at the **★ EXTENSION POINT ★** comment.
3. Add a `draw_pothole_mask()` helper in `visualization/renderer.py`.

### Add lane detection
1. Create `perception/lane_detector.py`.
2. In `filtering/region_filter.py` add a `lane_filter(detections, lane_mask)`.
3. Pass the steering offset from lane curvature into `VehicleController.step()`.

### Connect to real hardware (ROS / CAN)
1. Replace `utils/frame_loader.py`'s `iter_frames` with a ROS subscriber generator.
2. In `control/controller.py`, publish `VehicleState` fields to your CAN bus
   instead of (or in addition to) the simulation integration.

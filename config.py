"""
config.py — Central configuration for the autonomous driving pipeline.

All tunable constants live here so the rest of the codebase stays free
of magic numbers. Change values here to retune behaviour without touching
pipeline logic.
"""

# ---------------------------------------------------------------------------
# Hardware / runtime
# ---------------------------------------------------------------------------
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
TARGET_FPS   = 30          # used to derive dt; not a hard cap

# ---------------------------------------------------------------------------
# YOLO detection
# ---------------------------------------------------------------------------
YOLO_MODEL        = "yolov8n.pt"
YOLO_CONF_THRESH  = 0.50

# COCO class IDs we care about. Extend freely (e.g. add 15 = "bench").
RELEVANT_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Humans get a depth priority boost so they're never ignored if a vehicle
# happens to be slightly closer.
HUMAN_PRIORITY_BOOST = 0.10   # added to depth score for class 0

# ---------------------------------------------------------------------------
# MiDaS depth model
# ---------------------------------------------------------------------------
MIDAS_MODEL = "MiDaS_small"   # swap to "DPT_Large" for better quality

# ---------------------------------------------------------------------------
# Spatial region boundaries (fraction of frame width)
# ---------------------------------------------------------------------------
LEFT_BOUNDARY   = 0.33   # 0 … LEFT_BOUNDARY          → LEFT zone
RIGHT_BOUNDARY  = 0.66   # RIGHT_BOUNDARY … 1.0        → RIGHT zone
                          # in between                  → CENTER zone

# Only objects whose bounding-box centre y > this fraction of height are
# treated as "nearby" and eligible for braking decisions.
NEAR_HALF_THRESHOLD = 0.40   # lower = stricter; objects must be in bottom 60 %

# ---------------------------------------------------------------------------
# Depth / approaching detection
# ---------------------------------------------------------------------------
# MiDaS returns *inverse* depth (high = close). After normalising 0-1 a
# bigger value means the object is closer.
DEPTH_APPROACHING_THRESHOLD = 0.025  # delta that counts as "moving toward us"
DEPTH_SMOOTH_ALPHA          = 0.4    # EMA weight for per-object depth history

# ---------------------------------------------------------------------------
# Control thresholds
# ---------------------------------------------------------------------------
DEPTH_HARD_BRAKE = 0.70
DEPTH_BRAKE      = 0.50
DEPTH_SLOW       = 0.30

ACC_HARD_BRAKE   = -20.0
ACC_BRAKE        = -10.0
ACC_SLOW         =  -2.0
ACC_GO           =  +5.0
ACC_APPROACHING_PENALTY = -5.0   # extra deceleration when object is closing

# ---------------------------------------------------------------------------
# Vehicle / physics
# ---------------------------------------------------------------------------
INITIAL_VELOCITY = 60.0   # km/h (or arbitrary units – consistent throughout)
MAX_VELOCITY     = 80.0
MIN_VELOCITY     =  0.0
DT               = 0.1    # seconds per frame step

# Temporal smoothing — controls how quickly velocity & steering can change.
# Lower = smoother / slower reaction.  Higher = snappier.
#
# VELOCITY: EMA weight applied to the *target* velocity each frame.
#   0.05 = very gradual (ramp over ~20 frames)
#   0.20 = moderate   (ramp over ~5 frames)
VELOCITY_ALPHA   = 0.08

# STEERING: EMA weight (separate from velocity).
#   0.15 = smooth but responsive
STEERING_ALPHA   = 0.15

# Hard rate-limits (km/h per second).  Prevents the controller from
# applying unrealistically large accelerations in a single step.
# E.g. MAX_ACCEL_RATE = 8  → a car can gain at most 8 km/h per second.
MAX_ACCEL_RATE   =  8.0   # km/h per second  (positive = speeding up)
MAX_DECEL_RATE   = 20.0   # km/h per second  (positive magnitude = slowing down)

# How much of the previous *output* acceleration is blended into the new one.
# Gives acceleration itself temporal inertia (avoids step jumps in the HUD).
ACCEL_SMOOTH_ALPHA = 0.3  # 0 = no inertia, 1 = never changes

# ---------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------
STEER_MAX   =  1.0
STEER_MIN   = -1.0
STEER_STEP  =  0.3   # magnitude applied per frame when obstacle detected

# ---------------------------------------------------------------------------
# Visualisation colours  (BGR)
# ---------------------------------------------------------------------------
COLOR_PERSON    = (0,   255,  80)
COLOR_VEHICLE   = (0,   100, 255)
COLOR_APPROACH  = (0,   0,   255)
COLOR_REGION    = (180, 180,  50)
COLOR_HUD_TEXT  = (255, 255, 255)
COLOR_HUD_WARN  = (0,   0,   255)

# ---------------------------------------------------------------------------
# Road trapezoid (drivable area mask)
# ---------------------------------------------------------------------------
# A perspective trapezoid that approximates the road surface visible from
# a forward-facing camera mounted at bonnet / windscreen height.
#
# All values are FRACTIONS of frame width/height so they scale with any
# resolution.  Tune these by overlaying the mask on a sample KITTI frame
# (set ROAD_MASK_DEBUG = True in config) until the polygon hugs the road.
#
#   top-left        top-right
#       *──────────────*          ← ROAD_TOP_Y (e.g. 0.55 = 55 % down)
#      /                \
#     /                  \
#    *────────────────────*       ← bottom of frame (y = 1.0)
#  bottom-left        bottom-right
#
ROAD_TOP_Y        = 0.55   # horizon line  (fraction of height)
ROAD_TOP_LEFT_X   = 0.35   # top-left  x   (fraction of width)
ROAD_TOP_RIGHT_X  = 0.65   # top-right x
ROAD_BOT_LEFT_X   = 0.05   # bottom-left x
ROAD_BOT_RIGHT_X  = 0.95   # bottom-right x

# Fraction of a bounding box that must overlap the road mask for the
# detection to be "on-road".  With the segmentation-based RoadDetector the
# mask is semantically accurate so we can afford a stricter threshold.
# 0.40 = 40 % overlap required (up from 0.30).
ROAD_OVERLAP_THRESH = 0.40

# Set True to draw the road polygon on every frame (useful for tuning)
ROAD_MASK_DEBUG = True

# ---------------------------------------------------------------------------
# Lane detection (perception/lane_detector.py)
# ---------------------------------------------------------------------------
# Pre-processing
LANE_BLUR_KSIZE      = 7      # Gaussian blur kernel (must be odd)
LANE_CANNY_LOW       = 50     # lower Canny threshold
LANE_CANNY_HIGH      = 150    # upper Canny threshold

# Hough transform
LANE_HOUGH_THRESHOLD  = 30    # minimum votes to accept a line
LANE_MIN_LINE_LENGTH  = 30    # px — shorter segments are noise
LANE_MAX_LINE_GAP     = 80    # px — gaps within a dashed line

# Line classification
LANE_MIN_SLOPE        = 0.4   # |slope| below this = horizontal noise

# How much the car being off-centre matters for steering
# 1.0 = full steer correction per pixel of centre offset / (w/2)
LANE_CENTRE_STEER_GAIN = 0.6

# ---------------------------------------------------------------------------
# Obstacle count steering (control/controller.py)
# ---------------------------------------------------------------------------
# When all three regions (LEFT, CENTER, RIGHT) have on-road obstacles,
# steer toward the region with the fewest detections.
# Tie-break: prefer RIGHT (most roads drive on the left / right of centre).
OBSTACLE_TIEBREAK_PREFER_RIGHT = True

# ---------------------------------------------------------------------------
# Road Detector (perception/road_detector.py)
# ---------------------------------------------------------------------------
# Inference resolution for DeepLabV3. Smaller = faster, coarser mask.
ROAD_DETECTOR_W = 320
ROAD_DETECTOR_H = 320

# ROI: top fraction to zero-out in the road mask (Fix 5).
# 0.50 means only the bottom half of the frame is used as drivable area.
ROAD_ROI_TOP_FRACTION = 0.50


# ---------------------------------------------------------------------------
# LED ESP32
# ---------------------------------------------------------------------------
LED_ESP32_IP  = "http://espcam.local"   # ← replace with your ESP32 DevKit V1 IP
LED_DEAD_V     = 0.3             # ignore tiny velocity values
LED_DEAD_STEER = 0.05            # ignore tiny steer (cruise)
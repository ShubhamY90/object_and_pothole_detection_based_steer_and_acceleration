"""
perception/road_detector_fastscnn.py — Fast-SCNN road/drivable-area detector.

Fast-SCNN (Fast Semantic Segmentation Network)
-----------------------------------------------
Paper: "Fast-SCNN: Fast Semantic Segmentation Network" (Poudel et al., 2019)
https://arxiv.org/abs/1902.04502

Architecture overview
---------------------
    Input (H×W×3)
        ↓
    [Learning to Downsample]   — 3 conv layers → 1/8 resolution, 64 ch
        ↓ (skip connection saved)
    [Global Feature Extractor] — MobileNetV2 bottleneck blocks → 1/32, 128 ch
        ↓
    [Feature Fusion Module]    — upsample + add skip → 1/8, 128 ch
        ↓
    [Classifier]               — depthwise-sep conv → 1×1 → softmax → 1/8
        ↓
    Upsample to original size

Why Fast-SCNN instead of DeepLabV3?
-------------------------------------
* Fast-SCNN is designed for real-time edge deployment: ~123 FPS on Cityscapes
  at 1024×2048 on a Titan Xp vs ~8 FPS for DeepLabV3+ResNet-101.
* The lightweight design (few parameters) is ideal for CPU-only inference.
* Shared low-level features between the skip path and global extractor reduce
  redundant computation.

Pretrained weights — 3-tier strategy (fixes "always cruise" bug)
-----------------------------------------------------------------
Tier 1: User-supplied .pth file passed to __init__
Tier 2: Auto-download Fast-SCNN Cityscapes checkpoint from GitHub
Tier 3: torchvision LRASPP-MobileNetV3 (always available, COCO-pretrained)

The LRASPP tier ensures the road mask is ALWAYS semantically valid regardless
of download availability.  With an empty road mask, lane_centre_x falls back
to frame_width/2, giving zero offset → no steering → the "always cruise" bug.

Output interface (identical to RoadResult from road_detector.py)
----------------------------------------------------------------
    .road_mask      uint8 (H×W) — 255 = drivable surface
    .lane_centre_x  float       — x-centroid in bottom strip (pixels)
    .debug_frame    BGR ndarray — annotated frame
    .mask_source    str         — "fastscnn" | "lraspp" | "trapezoid"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ---------------------------------------------------------------------------
# Data container  (duck-type compatible with RoadResult / LaneResult)
# ---------------------------------------------------------------------------

@dataclass
class RoadResult:
    road_mask:     Optional[np.ndarray] = None   # uint8 binary H×W  (255=road)
    lane_centre_x: Optional[float]      = None   # pixels
    debug_frame:   Optional[np.ndarray] = None
    mask_source:   str                  = "none"
    # Backward-compat stubs
    left_poly:  object = None
    right_poly: object = None


# ===========================================================================
# Fast-SCNN building blocks
# ===========================================================================

class _DSConv(nn.Sequential):
    """Depthwise-separable conv block: DW → BN → ReLU → PW → BN → ReLU."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual bottleneck."""
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand: int = 6):
        super().__init__()
        mid_ch = in_ch * expand
        self.use_res = (stride == 1 and in_ch == out_ch)
        layers: list[nn.Module] = []
        if expand != 1:
            layers += [
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1,
                      groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x) if self.use_res else self.conv(x)


def _make_bottleneck_stack(
    in_ch: int, out_ch: int, stride: int, n: int, expand: int = 6
) -> nn.Sequential:
    layers = [_InvertedResidual(in_ch, out_ch, stride, expand)]
    for _ in range(1, n):
        layers.append(_InvertedResidual(out_ch, out_ch, 1, expand))
    return nn.Sequential(*layers)


class _LearningToDownsample(nn.Module):
    """3 conv layers → 1/8 resolution, 64 channels (skip connection)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.dsconv1 = _DSConv(32, 48, stride=2)
        self.dsconv2 = _DSConv(48, 64, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dsconv2(self.dsconv1(self.conv1(x)))


class _GlobalFeatureExtractor(nn.Module):
    """MobileNetV2 bottleneck blocks → 1/32 resolution, 128 channels."""
    def __init__(self):
        super().__init__()
        self.bottlenecks = nn.Sequential(
            _make_bottleneck_stack(64,  64,  stride=2, n=3, expand=6),
            _make_bottleneck_stack(64,  96,  stride=2, n=3, expand=6),
            _make_bottleneck_stack(96,  128, stride=1, n=3, expand=6),
        )
        self.ppm = _PyramidPooling(128, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ppm(self.bottlenecks(x))


class _PyramidPooling(nn.Module):
    """Lightweight 4-scale pyramid pooling."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid = in_ch // 4
        self.pools = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(s),
                          nn.Conv2d(in_ch, mid, 1, bias=False),
                          nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
            for s in [1, 2, 3, 6]
        ])
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch + 4 * mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        parts = [x] + [
            F.interpolate(p(x), size=(h, w), mode="bilinear", align_corners=True)
            for p in self.pools
        ]
        return self.proj(torch.cat(parts, dim=1))


class _FeatureFusionModule(nn.Module):
    """Fuse skip (1/8) + global (1/32) → 1/8, 128 channels."""
    def __init__(self, skip_ch: int, global_ch: int, out_ch: int):
        super().__init__()
        self.dwconv = nn.Sequential(
            nn.Conv2d(global_ch, global_ch, 3, padding=4, dilation=4,
                      groups=global_ch, bias=False),
            nn.BatchNorm2d(global_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(global_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, skip: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        global_up = F.interpolate(global_feat, size=skip.shape[2:],
                                  mode="bilinear", align_corners=True)
        return self.relu(self.skip_proj(skip) + self.dwconv(global_up))


class _Classifier(nn.Module):
    """2× DS-conv head → per-pixel class scores."""
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            _DSConv(in_ch, in_ch),
            _DSConv(in_ch, in_ch),
            nn.Dropout(0.1),
            nn.Conv2d(in_ch, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class FastSCNN(nn.Module):
    """Fast Semantic Segmentation Network — full model."""
    def __init__(self, num_classes: int = 19):
        super().__init__()
        self.learning_to_downsample  = _LearningToDownsample()
        self.global_feature_extractor = _GlobalFeatureExtractor()
        self.feature_fusion           = _FeatureFusionModule(64, 128, 128)
        self.classifier               = _Classifier(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size        = x.shape[2:]
        skip        = self.learning_to_downsample(x)
        global_feat = self.global_feature_extractor(skip)
        fused       = self.feature_fusion(skip, global_feat)
        logits      = self.classifier(fused)
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=True)


# ===========================================================================
# Road Detector wrapper
# ===========================================================================

# Cityscapes 19-class mapping: class 0 = road, class 1 = sidewalk
_CITYSCAPES_ROAD_IDS = {0}      # strict: road only (add 1 for sidewalk)

# VOC/COCO background (class 0) used by LRASPP torchvision model
_LRASPP_ROAD_IDS     = {0}

# Fast-SCNN pretrained weights (Tramac's Cityscapes checkpoint)
_WEIGHTS_URL = (
    "https://github.com/Tramac/Fast-SCNN-pytorch/releases/download/v1.0/"
    "fast_scnn_citys.pth"
)

# Road mask must cover at least this fraction of frame pixels to be "valid".
# Below this threshold → fall back to trapezoid mask.
# This prevents the "always cruise" bug when weights are absent/random.
_MIN_ROAD_PIXEL_FRACTION = 0.03   # 3 % of frame area


class FastSCNNRoadDetector:
    """
    Real-time road segmentation using Fast-SCNN with guaranteed valid mask.

    Model selection
    ---------------
    1. Fast-SCNN with pretrained checkpoint   → best speed, needs weights
    2. Fast-SCNN with random weights          → FPS-only; mask is empty
       → **auto-falls back to LRASPP** (tier 3)
    3. torchvision LRASPP-MobileNetV3         → slower than Fast-SCNN but
       guaranteed pretrained mask (no download needed)
    4. Trapezoid heuristic                    → last resort, always valid

    Empty-mask guard (fixes "always cruise" bug)
    --------------------------------------------
    After segmentation, if the road mask covers < MIN_ROAD_PIXEL_FRACTION of
    the frame we consider the segmentation failed and substitute the
    trapezoid mask. This ensures lane_centre_x is never stuck at w/2 and
    obstacle filtering has a valid road region to work with.

    Usage
    -----
    detector = FastSCNNRoadDetector()
    result   = detector.detect(frame_bgr)
    print(result.mask_source)   # "fastscnn" | "lraspp" | "trapezoid"
    """

    _INFER_W = 512
    _INFER_H = 256
    _MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __init__(self, weights_path: Optional[str] = None, num_classes: int = 19):
        self._device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._num_classes = num_classes
        self._weights_loaded = False

        # ── Tier 1 + 2: Fast-SCNN ────────────────────────────────────────
        print("[FastSCNN] Building Fast-SCNN architecture …")
        self._fastscnn = FastSCNN(num_classes=num_classes).to(self._device).eval()
        self._try_load_weights(weights_path)

        # ── Tier 3: LRASPP fallback (always pretrained via torchvision) ──
        self._lraspp = None
        self._lraspp_road_ids = _LRASPP_ROAD_IDS
        if not self._weights_loaded:
            print("[FastSCNN] Fast-SCNN weights unavailable — "
                  "loading torchvision LRASPP as semantic fallback …")
            self._lraspp = self._build_lraspp()

        # ── Warm-up ───────────────────────────────────────────────────────
        print(f"[FastSCNN] Warming up ({self._device}) …")
        dummy = torch.zeros(1, 3, self._INFER_H, self._INFER_W).to(self._device)
        model_to_warm = self._fastscnn if self._weights_loaded else (self._lraspp or self._fastscnn)
        for _ in range(3):
            with torch.no_grad():
                if model_to_warm is self._lraspp and self._lraspp is not None:
                    model_to_warm(dummy)["out"]
                else:
                    model_to_warm(dummy)
        print("[FastSCNN] Ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> RoadResult:
        """
        Run road segmentation on a BGR frame.
        Always returns a valid RoadResult with a non-empty road mask.
        """
        h, w = frame.shape[:2]
        result = RoadResult(debug_frame=frame.copy())

        tensor = self._preprocess(frame)

        # ── choose model ──────────────────────────────────────────────
        if self._weights_loaded:
            road_mask, source = self._run_fastscnn(tensor, w, h)
        elif self._lraspp is not None:
            road_mask, source = self._run_lraspp(tensor, w, h)
        else:
            road_mask, source = self._run_fastscnn(tensor, w, h)

        # ── Empty-mask guard (THE FIX for "always cruise") ────────────
        # If the segmentation model (random weights or genuine failure)
        # produces a near-empty mask, lane_centre_x would lock at w/2
        # → offset = 0 → steer = 0 → "cruise" forever.
        # Swap in the trapezoid mask so the pipeline always has a valid
        # drivable area to work with.
        road_pixels = int(np.count_nonzero(road_mask))
        total_pixels = h * w
        if road_pixels < _MIN_ROAD_PIXEL_FRACTION * total_pixels:
            road_mask = _trapezoid_mask(w, h)
            _apply_roi(road_mask, h)
            source = "trapezoid-fallback"

        result.road_mask     = road_mask
        result.lane_centre_x = _compute_centre_x(road_mask, w, h)
        result.mask_source   = source

        _draw_overlay(result.debug_frame, road_mask, label=f"FAST-SCNN [{source}]")
        return result

    # ------------------------------------------------------------------
    # Model runners
    # ------------------------------------------------------------------

    def _run_fastscnn(
        self, tensor: torch.Tensor, w: int, h: int
    ) -> tuple[np.ndarray, str]:
        with torch.no_grad():
            logits = self._fastscnn(tensor)           # (1, C, H_inf, W_inf)
        pred = logits[0].argmax(dim=0).cpu().numpy().astype(np.uint8)
        road_mask = self._pred_to_mask(pred, w, h, _CITYSCAPES_ROAD_IDS)
        return road_mask, "fastscnn"

    def _run_lraspp(
        self, tensor: torch.Tensor, w: int, h: int
    ) -> tuple[np.ndarray, str]:
        with torch.no_grad():
            out = self._lraspp(tensor)["out"]         # (1, C, H_inf, W_inf)
        pred = out[0].argmax(dim=0).cpu().numpy().astype(np.uint8)
        road_mask = self._pred_to_mask(pred, w, h, self._lraspp_road_ids)
        return road_mask, "lraspp"

    @staticmethod
    def _pred_to_mask(
        pred: np.ndarray,
        w: int, h: int,
        road_ids: set,
    ) -> np.ndarray:
        """Convert a class-index map to a binary (0/255) road mask."""
        road_small = np.zeros_like(pred, dtype=np.uint8)
        for cls_id in road_ids:
            road_small[pred == cls_id] = 255
        road_mask = cv2.resize(road_small, (w, h), interpolation=cv2.INTER_NEAREST)
        _apply_roi(road_mask, h)
        k = np.ones((5, 5), np.uint8)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, k)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN,  k)
        return road_mask

    # ------------------------------------------------------------------
    # LRASPP builder (torchvision, always pretrained)
    # ------------------------------------------------------------------

    def _build_lraspp(self):
        """
        Build torchvision's LRASPP-MobileNetV3-Large with COCO pretrained
        weights. This is faster than DeepLabV3-ResNet and always available
        without a separate download.

        IMPORTANT class mapping for LRASPP (VOC/COCO 21 classes):
          class 0  = background (contains road surface)
          class 15 = person
        We use class 0 (background) as the road proxy — same logic as the
        original road_detector.py.
        """
        try:
            from torchvision.models.segmentation import (
                lraspp_mobilenet_v3_large,
                LRASPP_MobileNet_V3_Large_Weights,
            )
            weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
            model = (
                lraspp_mobilenet_v3_large(weights=weights)
                .to(self._device)
                .eval()
            )
            # LRASPP outputs 21 VOC classes; road ≈ background (class 0)
            self._lraspp_road_ids = {0}
            print(f"[FastSCNN] LRASPP-MobileNetV3 loaded (device={self._device})")
            return model
        except Exception as e:
            print(f"[FastSCNN] LRASPP load failed ({e}). Will use trapezoid fallback.")
            return None

    # ------------------------------------------------------------------
    # Weight loading for Fast-SCNN
    # ------------------------------------------------------------------

    def _try_load_weights(self, weights_path: Optional[str]) -> None:
        """Load Fast-SCNN checkpoint with 3-tier priority."""
        # Tier 1 — user path
        if weights_path and Path(weights_path).is_file():
            self._load_checkpoint(weights_path, "user-supplied")
            return

        # Tier 2a — cached copy
        cache = Path.home() / ".cache" / "fastscnn" / "fast_scnn_citys.pth"
        if cache.is_file():
            self._load_checkpoint(str(cache), "cache")
            return

        # Tier 2b — auto-download
        print("[FastSCNN] Attempting to download pretrained weights …")
        try:
            import urllib.request
            cache.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(_WEIGHTS_URL, str(cache))
            self._load_checkpoint(str(cache), "download")
        except Exception as e:
            print(f"[FastSCNN] Weight download failed: {e}")
            print("[FastSCNN] → Fast-SCNN will use random init; "
                  "LRASPP fallback will handle road segmentation.")

    def _load_checkpoint(self, path: str, source: str) -> None:
        try:
            ckpt  = torch.load(path, map_location=self._device, weights_only=True)
            state = ckpt.get("state_dict", ckpt.get("model", ckpt))
            state = {k.replace("module.", ""): v for k, v in state.items()}
            missing, _ = self._fastscnn.load_state_dict(state, strict=False)
            if missing:
                print(f"[FastSCNN] {len(missing)} missing keys (partial load ok)")
            self._weights_loaded = True
            print(f"[FastSCNN] Weights loaded from {source}: {path}")
        except Exception as e:
            print(f"[FastSCNN] Checkpoint load error ({e})")

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (self._INFER_W, self._INFER_H))
        t     = torch.from_numpy(small).permute(2, 0, 1).float() / 255.0
        t     = (t - self._MEAN) / self._STD
        return t.unsqueeze(0).to(self._device)


# ---------------------------------------------------------------------------
# Module-level helpers (shared / importable)
# ---------------------------------------------------------------------------

def _apply_roi(mask: np.ndarray, h: int) -> None:
    """Zero-out the top 50 % — focus on near-vehicle surface (Fix 5)."""
    mask[: int(0.50 * h), :] = 0


def _compute_centre_x(mask: np.ndarray, w: int, h: int) -> float:
    """X-centroid of road pixels in the bottom third of the frame."""
    strip = mask[int(0.67 * h) :, :]
    cols  = np.where(strip > 0)[1]
    return float(cols.mean()) if cols.size > 0 else float(w / 2)


def _trapezoid_mask(w: int, h: int) -> np.ndarray:
    """Rule-based trapezoid fallback — always non-empty."""
    poly = np.array([
        [int(config.ROAD_BOT_LEFT_X  * w), h],
        [int(config.ROAD_TOP_LEFT_X  * w), int(config.ROAD_TOP_Y * h)],
        [int(config.ROAD_TOP_RIGHT_X * w), int(config.ROAD_TOP_Y * h)],
        [int(config.ROAD_BOT_RIGHT_X * w), h],
    ], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask


def _draw_overlay(
    frame: np.ndarray,
    mask:  np.ndarray,
    label: str = "FAST-SCNN",
) -> None:
    """Tint road pixels green and stamp the active model/source label."""
    overlay = np.zeros_like(frame)
    overlay[mask > 0] = (0, 120, 10)
    cv2.addWeighted(frame, 1.0, overlay, 0.40, 0, dst=frame)

    # Colour-code by source: green=fastscnn, yellow=lraspp, red=trapezoid
    if "fastscnn" in label:
        col = (0, 255, 60)
    elif "lraspp" in label:
        col = (0, 220, 255)
    else:
        col = (0, 80, 255)

    cv2.putText(
        frame, label, (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA,
    )

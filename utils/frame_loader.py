"""
utils/frame_loader.py — Thin wrapper around cv2 for loading image sequences.

Keeping I/O in one place means you can swap KITTI for a webcam stream or
a ROS topic by touching only this file.
"""

import os
from typing import Generator, Tuple
import cv2
import numpy as np

import config


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_frames(
    folder:     str,
    width:      int = config.FRAME_WIDTH,
    height:     int = config.FRAME_HEIGHT,
) -> Generator[Tuple[str, np.ndarray], None, None]:
    """
    Yield (filename, resized_bgr_frame) for every image in *folder*,
    sorted lexicographically (KITTI naming is zero-padded so this is
    equivalent to temporal order).

    Frames that fail to load are silently skipped with a warning.
    """
    files = sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        raise FileNotFoundError(
            f"No supported images found in '{folder}'. "
            f"Expected one of {SUPPORTED_EXTENSIONS}"
        )

    for filename in files:
        path  = os.path.join(folder, filename)
        frame = cv2.imread(path)

        if frame is None:
            print(f"[WARN] Could not read '{path}' – skipping.")
            continue

        frame = cv2.resize(frame, (width, height))
        yield filename, frame


def webcam_stream(
    device_id:  int = 0,
    width:      int = config.FRAME_WIDTH,
    height:     int = config.FRAME_HEIGHT,
) -> Generator[Tuple[str, np.ndarray], None, None]:
    """
    Yield ("webcam", frame) from a live USB / built-in camera.
    Mirrors the same interface as iter_frames so the main loop needs
    no changes when switching input sources.
    """
    cap = cv2.VideoCapture(device_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield "webcam", frame
    finally:
        cap.release()

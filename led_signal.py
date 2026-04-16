"""
led_signal.py — Send PWM brightness + directional signals to ESP32-CAM via HTTP.

PWM channels
------------
Green LED  (GPIO 18) = forward speed
Red   LED  (GPIO 19) = turning magnitude OR reversing

Direction pins (digital HIGH / LOW)
------------------------------------
GPIO 12 (UP)    = vehicle commanded forward
GPIO 13 (DOWN)  = vehicle commanded in reverse
GPIO 14 (LEFT)  = left-turn commanded
GPIO 15 (RIGHT) = right-turn commanded

Steering sign convention: negative = left, positive = right.
Flip the signs in _compute_direction_pins() if your controller differs.

Dataset logging
---------------
Every call appends one CSV row to config.SIGNAL_LOG_CSV.
Columns: timestamp, velocity, steering, pwm_green, pwm_red,
         gpio12_up, gpio13_down, gpio14_left, gpio15_right
"""

import csv
import os
import time

import requests
import config

# ── Module-level state ──────────────────────────────────────────────────────
_csv_header_written: bool = False


# ── Dataset logger ───────────────────────────────────────────────────────────

def _log_to_dataset(
    velocity: float,
    steering: float,
    g:        int,
    r:        int,
    up:       int,
    down:     int,
    left:     int,
    right:    int,
) -> None:
    """
    Append one row to the signal dataset CSV.
    Creates the file (and parent directories) if they do not exist.
    Header is written only once per process lifetime.
    """
    global _csv_header_written

    log_path   = getattr(config, "SIGNAL_LOG_CSV", "signal_log.csv")
    parent_dir = os.path.dirname(os.path.abspath(log_path))
    os.makedirs(parent_dir, exist_ok=True)

    write_header = (not _csv_header_written) and (not os.path.exists(log_path))

    with open(log_path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow([
                "timestamp",
                "velocity",
                "steering",
                "pwm_green",
                "pwm_red",
                "gpio12_up",
                "gpio13_down",
                "gpio14_left",
                "gpio15_right",
            ])
        writer.writerow([
            f"{time.time():.4f}",
            f"{velocity:.4f}",
            f"{steering:.4f}",
            g,
            r,
            up,
            down,
            left,
            right,
        ])

    _csv_header_written = True


# ── Direction logic ───────────────────────────────────────────────────────────

def _compute_direction_pins(
    velocity: float,
    steering: float,
) -> tuple[int, int, int, int]:
    """
    Map velocity / steering floats to digital direction pin values.

    Returns
    -------
    (up, down, left, right) where each value is 0 or 1.

    GPIO 12 UP    — set HIGH when moving forward
    GPIO 13 DOWN  — set HIGH when reversing
    GPIO 14 LEFT  — set HIGH when steering left  (steering < -dead_steer)
    GPIO 15 RIGHT — set HIGH when steering right (steering >  dead_steer)

    Only one of UP / DOWN can be 1 at a time.
    LEFT and RIGHT are mutually exclusive by the dead-band logic.
    """
    dead_v     = getattr(config, "LED_DEAD_V",     0.05)
    dead_steer = getattr(config, "LED_DEAD_STEER", 0.05)

    up    = 1 if velocity  >  dead_v     else 0
    down  = 1 if velocity  < -dead_v     else 0
    left  = 1 if steering  < -dead_steer else 0
    right = 1 if steering  >  dead_steer else 0

    return up, down, left, right


# ── Public API ────────────────────────────────────────────────────────────────

def send_leds(velocity: float, steering: float) -> None:
    """
    Compute PWM values + direction pins → send to ESP32-CAM → log to CSV.

    This function is intentionally non-blocking:
    - HTTP timeout is 0.1 s; any network error is silently swallowed.
    - CSV I/O is synchronous but cheap (one row append).

    Parameters
    ----------
    velocity : float
        Signed vehicle speed in the normalised range [-MAX_VELOCITY, +MAX_VELOCITY].
        Positive = forward, negative = reverse.
    steering : float
        Signed steering command in [-1.0, +1.0].
        Negative = left, positive = right.
    """
    dead_v     = getattr(config, "LED_DEAD_V",     0.05)
    max_vel    = getattr(config, "MAX_VELOCITY",    1.0)
    dead_steer = getattr(config, "LED_DEAD_STEER", 0.05)
    esp_ip     = getattr(config, "LED_ESP32_IP",   "http://192.168.4.1")

    # ── PWM: green (forward speed) ────────────────────────────────────────
    if velocity > dead_v:
        g = int(min(velocity / max_vel, 1.0) * 255)
    else:
        g = 0

    # ── PWM: red (turning OR reversing — whichever is stronger) ──────────
    steer_r = (
        int(min(abs(steering), 1.0) * 255)
        if abs(steering) > dead_steer
        else 0
    )
    back_r = (
        int(min(-velocity / max_vel, 1.0) * 255)
        if velocity < -dead_v
        else 0
    )
    r = max(steer_r, back_r)

    # Idle glow — confirms the system is alive when completely stationary
    if g == 0 and r == 0:
        g = 15

    # ── Direction pins ────────────────────────────────────────────────────
    up, down, left, right = _compute_direction_pins(velocity, steering)

    # ── HTTP: send everything in one GET request ──────────────────────────
    # [COMMENTED OUT] to prevent DNS blocking (FPS drop) when ESP32 is offline
    # try:
    #     requests.get(
    #         f"{esp_ip}/led",
    #         params={
    #             "g":     g,
    #             "r":     r,
    #             "up":    up,     # → GPIO 12
    #             "down":  down,   # → GPIO 13
    #             "left":  left,   # → GPIO 14
    #             "right": right,  # → GPIO 15
    #         },
    #         timeout=0.1,
    #     )
    # except requests.exceptions.RequestException:
    #     pass   # never stall the driving loop

    # ── Dataset: log every call ───────────────────────────────────────────
    _log_to_dataset(velocity, steering, g, r, up, down, left, right)
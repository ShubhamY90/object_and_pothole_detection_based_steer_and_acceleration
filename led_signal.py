"""
led_signal.py — Send PWM brightness values to ESP32 LED controller via HTTP.

Green LED = forward motion  (GPIO 18)
Red   LED = turning or reversing (GPIO 19)
"""

import requests
import config

def send_leds(velocity: float, steering: float) -> None:
    # Green: forward speed
    if velocity > config.LED_DEAD_V:
        g = int(min(velocity / config.MAX_VELOCITY, 1.0) * 255)
    else:
        g = 0

    # Red: turning magnitude OR reversing — take whichever is stronger
    steer_r = int(min(abs(steering), 1.0) * 255) if abs(steering) > config.LED_DEAD_STEER else 0
    back_r  = int(min(-velocity / config.MAX_VELOCITY, 1.0) * 255) if velocity < -config.LED_DEAD_V else 0
    r = max(steer_r, back_r)

    # idle glow so you know the system is alive
    if g == 0 and r == 0:
        g = 15

    try:
        requests.get(
            f"{config.LED_ESP32_IP}/led",
            params={"g": g, "r": r},
            timeout=0.1,   # non-blocking — never stall the driving loop
        )
    except requests.exceptions.RequestException:
        pass   # silently skip if ESP32 unreachable
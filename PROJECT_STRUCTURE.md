# Autonomous Driving & ESP32-CAM Pipeline: Project Structure

This document breaks down every folder and file in the project, explaining **what it is**, **why it is there**, and **how they all fit together**.

---

## 1. Main Entry Points (The Executables)

These are the core scripts you run to start the pipeline. They all use the same underlying brain (YOLO, filtering, control logic) but get their video from different sources.

* **`main1.py` & `main2.py`**
  * **What:** Earlier versions of the pipeline.
  * **Why:** Likely used for testing initial YOLO and steering math on local MP4 videos before the Fast-SCNN (road segmentation) or Hardware (ESP32) pieces were mature.
* **`main3.py`** (The Demo / Simulator)
  * **What:** Pulls video frames from a saved local dataset/video (in `data/`) but treats it as if it's the real world. 
  * **Why:** Used for testing your logic securely at your desk. It processes the fake video, decides how to steer, and **sends the LED signals to the physical ESP32** on your desk so you can verify the hardware wiring without leaving your room.
* **`main4.py`** (The Live Production Code)
  * **What:** The real deal. Connects to `http://<ESP_IP>/stream`, pulls real live video from the miniature car, processes the frames, and shoots steering/velocity commands straight back to the ESP32. 
  * **Why:** You run this when the car is placed on the road!

## 2. Hardware Utilities

Small scripts dedicated entirely to interacting with or testing the ESP32-CAM hardware.

* **`esp32cam_controller.ino`**
  * **What:** The master C++ firmware running on your ESP32-CAM microcontroller.
  * **Why:** It hosts an HTTP Server on Port `80` that constantly broadcasts a live MJPEG stream (`/stream`) while simultaneously listening for steering commands (`/led?up=1&left=1`) to physically turn on your wired LEDs.
* **`test_cam.py`**
  * **What:** A pure camera test script.
  * **Why:** To verify your Wi-Fi, IP address, and frame rate without loading heavy AI models. Just pure video streaming.
* **`collect_data.py`**
  * **What:** Connects to the ESP live stream and saves frames into a folder (`data/collected/`).
  * **Why:** If you want to train a custom YOLO model or Pothole detector later, you need pictures from the car's perspective. Runs fast and gathers data easily.
* **`test_leds.ino` & `wifi_led_only.ino`**
  * **What:** Old hardware testing sketches.
  * **Why:** Used originally to make sure LEDs were wired properly before building the complex stream code.

## 3. The "Brain" (Directories)

These folders constitute the actual autonomous logic. `main3.py` and `main4.py` just pass video into these folders and wait for a steering command to pop out.

* **`perception/`** (What do I see?)
  * `detector.py`: Holds the YOLOv8 AI. Looks at an image and returns bounding boxes (cars, people).
  * `road_detector_fastscnn.py`: AI that calculates exactly which pixels are drivable road vs. sidewalk/grass.
  * `pothole_detector.py`: Scans the drivable area specifically for hazards or potholes.
* **`filtering/`** (Should I care?)
  * Filters out objects YOLO found that aren't on the road, or are too far away to matter.
* **`tracking/`** (How fast is it moving?)
  * Associates objects frame-to-frame (e.g., establishing that the car 10 meters away is the same car from the last frame and calculating if it's getting closer).
* **`decision/`** (What should I do?)
  * Uses data from the trackers to decide if the car needs to brake (an object is approaching), swerve to avoid a pothole, or stay in its lane.
* **`control/`** (How do I maneuver?)
  * Translates the "decision" into raw physics. E.g., translates "Swerve Left" into "Steering = -0.5, Velocity = 10km/h".
* **`visualization/`** (Show me what you're thinking)
  * `renderer.py`: Draws the bounding boxes, the green road masks, and the text overlay on the screen so you can debug the AI's internal state visually.

## 4. Configuration & Logging

* **`config.py`**
  * **What:** The central control panel for the whole codebase.
  * **Why:** Instead of hunting through 10 different files to change the Maximum Speed, YOLO threshold, or the ESP32's IP address, you just change the variable in this single file.
* **`led_signal.py`**
  * **What:** Packages the output numbers from `control/` into an HTTP request and fires it at the ESP32.
  * **Why:** Also logs every single physical command to `signal_log.csv` so you can graph exactly when the car applied its brakes over time.
* **`requirements.txt`**
  * List of pip dependencies required to make it run (OpenCV, Torch, Ultralytics YOLO, etc).

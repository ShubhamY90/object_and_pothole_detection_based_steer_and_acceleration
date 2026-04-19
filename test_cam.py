import cv2
import argparse
import time
import requests

def main():
    parser = argparse.ArgumentParser(description="Test live video feed from ESP32-CAM")
    parser.add_argument("--esp-ip", default="10.46.37.132", 
                        help="The base IP of your ESP32-CAM (without http/port), e.g. 10.46.37.132")
    args = parser.parse_args()

    base_ip = args.esp_ip
    if not base_ip.startswith("http://"):
        base_ip = f"http://{base_ip}"

    print(f"--- Starting ESP32-CAM Test ---")

    # 2. Connect to Stream
    stream_url = f"{base_ip}/stream"
    print(f"[INFO] Connecting to MJPEG stream: {stream_url}")
    
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video stream at {stream_url}.")
        print("Please check your Wi-Fi, ensure the ESP32 is powered, and verify the IP address.")
        return

    print("[INFO] Stream started successfully. Press ESC to quit.")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[WARN] Dropped frame or stream ended.")
            break

        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0.0

        # Draw FPS on the screen
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("ESP32-CAM Live Test", frame)

        # Press ESC to exit
        if cv2.waitKey(1) == 27:
            print("[INFO] ESC pressed. Exiting...")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("--- Test Complete ---")

if __name__ == "__main__":
    main()

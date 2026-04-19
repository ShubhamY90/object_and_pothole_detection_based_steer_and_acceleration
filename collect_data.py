import argparse
import os
import time
import cv2
import requests

def main():
    parser = argparse.ArgumentParser(description="Collect images from ESP32-CAM and store them.")
    parser.add_argument("--esp-ip", default="10.46.37.132", help="Base IP of ESP32 (without http:// or port)")
    parser.add_argument("--out-dir", default="data/collected", help="Directory to save images")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between image captures in seconds")
    parser.add_argument("--max-frames", type=int, default=1000, help="Maximum number of frames to collect")
    args = parser.parse_args()

    base_ip = args.esp_ip
    if not base_ip.startswith("http://"):
        base_ip = f"http://{base_ip}"

    os.makedirs(args.out_dir, exist_ok=True)
    
    stream_url = f"{base_ip}/stream"
    print(f"[INFO] Connecting to MJPEG stream {stream_url} ...")
    cap = cv2.VideoCapture(stream_url)
    
    if not cap.isOpened():
        print("[ERROR] Failed to open stream.")
        return
        
    print(f"[INFO] Collecting up to {args.max_frames} frames into {args.out_dir} ...")
    
    try:
        for i in range(args.max_frames):
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Dropped frame from stream")
                continue
                
            filename = os.path.join(args.out_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            
            # Show the frame to user
            cv2.imshow("Data Collection", frame)
            if cv2.waitKey(1) == 27: # ESC
                break
                
            time.sleep(args.delay)
            
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Collection complete.")

if __name__ == "__main__":
    main()

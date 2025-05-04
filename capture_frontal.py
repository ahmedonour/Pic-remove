import cv2
import numpy as np
import mediapipe as mp
from rembg import remove
from datetime import datetime
import os

# --- Configuration ---
MAX_YAW_DEG = 10
MAX_PITCH_DEG = 10
OUTPUT_DIR = "captures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TEST_IMAGE_PATH = "test_image.jpg"  # Add a test image to your workspace

# --- MediaPipe setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# (Keep the MODEL_POINTS_3D, LANDMARK_IDX, and estimate_head_pose functions same as before)

def remove_bg_to_white(frame_rgb):
    result = remove(frame_rgb)
    rgb = cv2.cvtColor(result, cv2.COLOR_RGBA2RGB)
    mask = (result[:, :, 3] == 0)
    rgb[mask] = (255, 255, 255)
    return rgb


class CodespacesCapture:
    def __init__(self):
        self.running = False
        self.test_image = None
        
        # Try to load a test image if camera fails
        if os.path.exists(TEST_IMAGE_PATH):
            self.test_image = cv2.imread(TEST_IMAGE_PATH)
            print(f"Using test image: {TEST_IMAGE_PATH}")
        else:
            print("No camera available and no test image found")
            print("Create a 'test_image.jpg' or check camera permissions")

    def start(self):
        self.running = True
        print("Starting capture simulation... Press CTRL+C to stop")
        self.capture_loop()

    def capture_loop(self):
        while self.running:
            # Simulate camera feed
            if self.test_image is not None:
                frame = self.test_image.copy()
            else:
                # Create a synthetic frame
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "NO CAMERA INPUT", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Rest of your processing logic
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            fm_res = face_mesh.process(rgb)
            
            if fm_res.multi_face_landmarks:
                try:
                    lm = fm_res.multi_face_landmarks[0].landmark
                    # (Keep the rest of your pose estimation logic here)
                    # For demonstration, always "capture"
                    out = remove_bg_to_white(frame)
                    fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".png"
                    cv2.imwrite(os.path.join(OUTPUT_DIR, fname), out)
                    print(f"Simulated capture: {fname}")
                    
                except Exception as e:
                    print(f"Processing error: {str(e)}")

            # Slow down the loop for demonstration
            cv2.waitKey(1000)

if __name__ == "__main__":
    try:
        capture = CodespacesCapture()
        capture.start()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {str(e)}")
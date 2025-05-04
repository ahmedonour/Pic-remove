# Modified headless version
import cv2
import numpy as np
import mediapipe as mp
from rembg import remove
from datetime import datetime
import os

# --- Configuration --- (same as before)

class HeadlessCapture:
    def __init__(self):
        self.running = False

    def start(self):
        self.running = True
        self.capture_loop()

    def stop(self):
        self.running = False

    def capture_loop(self):
        # (rest of your capture_loop code without cv2.imshow)
        while self.running:
            # ... existing processing code ...
            
            # Remove this line:
            # cv2.imshow("Webcam", frame)
            
            # Replace ESC check with:
            key = cv2.waitKey(1)
            if key == 27:  # ESC still works
                self.stop()

if __name__ == "__main__":
    app = HeadlessCapture()
    print("Press Enter to start, ESC to stop")
    input()  # Wait for Enter key
    app.start()
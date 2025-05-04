import cv2
import threading
import numpy as np
import mediapipe as mp
from rembg import remove
from datetime import datetime
import os
import tkinter as tk
from tkinter import messagebox

# --- Configuration ---
MAX_YAW_DEG = 10
MAX_PITCH_DEG = 10
OUTPUT_DIR = "captures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- MediaPipe setup ---
mp_face_mesh = mp.solutions.face_mesh
mp_iris = mp.solutions.iris
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
iris = mp_iris.Iris(refine_landmarks=True)

MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1),
], dtype=np.float64)

LANDMARK_IDX = {
    "nose_tip": 1,
    "chin": 199,
    "left_eye": 33,
    "right_eye": 263,
    "mouth_left": 61,
    "mouth_right": 291
}

def estimate_head_pose(landmarks, image_shape):
    h, w = image_shape
    image_points = np.array([
        (landmarks[LANDMARK_IDX["nose_tip"]].x * w, landmarks[LANDMARK_IDX["nose_tip"]].y * h),
        (landmarks[LANDMARK_IDX["chin"]].x * w, landmarks[LANDMARK_IDX["chin"]].y * h),
        (landmarks[LANDMARK_IDX["left_eye"]].x * w, landmarks[LANDMARK_IDX["left_eye"]].y * h),
        (landmarks[LANDMARK_IDX["right_eye"]].x * w, landmarks[LANDMARK_IDX["right_eye"]].y * h),
        (landmarks[LANDMARK_IDX["mouth_left"]].x * w, landmarks[LANDMARK_IDX["mouth_left"]].y * h),
        (landmarks[LANDMARK_IDX["mouth_right"]].x * w, landmarks[LANDMARK_IDX["mouth_right"]].y * h),
    ], dtype=np.float64)

    cam_matrix = np.array([
        [w, 0, w / 2],
        [0, w, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rot_vec, trans_vec = cv2.solvePnP(MODEL_POINTS_3D, image_points, cam_matrix, dist_coeffs)
    if not success:
        return None

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    pose_mat = cv2.hconcat((rot_mat, trans_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    yaw, pitch, _ = euler_angles.flatten()
    return yaw, pitch

def remove_bg_to_white(frame_rgb):
    result = remove(frame_rgb)
    rgb = cv2.cvtColor(result, cv2.COLOR_RGBA2RGB)
    mask = (result[:, :, 3] == 0)
    rgb[mask] = (255, 255, 255)
    return rgb

class FaceCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face & Eye Tracker")
        self.running = False
        self.status_var = tk.StringVar()
        self.status_var.set("Status: Idle")

        tk.Button(root, text="Start Capture", command=self.start).pack(pady=5)
        tk.Button(root, text="Stop Capture", command=self.stop).pack(pady=5)
        tk.Label(root, textvariable=self.status_var).pack(pady=5)

    def start(self):
        if not self.running:
            self.running = True
            threading.Thread(target=self.capture_loop, daemon=True).start()
            self.status_var.set("Status: Running")

    def stop(self):
        self.running = False
        self.status_var.set("Status: Stopped")

    def capture_loop(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            fm_res = face_mesh.process(rgb)
            ir_res = iris.process(rgb)

            if fm_res.multi_face_landmarks and ir_res.multi_face_landmarks:
                lm = fm_res.multi_face_landmarks[0].landmark
                ir = ir_res.multi_face_landmarks[0].landmark

                head_pose = estimate_head_pose(lm, (h, w))
                if head_pose:
                    yaw, pitch = head_pose
                    li = ir[468]
                    ri = ir[473]
                    left_off = abs(li.x - lm[33].x) + abs(li.y - lm[33].y)
                    right_off = abs(ri.x - lm[263].x) + abs(ri.y - lm[263].y)

                    if abs(yaw) < MAX_YAW_DEG and abs(pitch) < MAX_PITCH_DEG and left_off < 0.03 and right_off < 0.03:
                        out = remove_bg_to_white(frame)
                        fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".png"
                        cv2.imwrite(os.path.join(OUTPUT_DIR, fname), out)
                        self.status_var.set(f"Captured: {fname}")
                        cv2.waitKey(1000)

            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) == 27:  # ESC to quit
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceCaptureApp(root)
    root.mainloop()

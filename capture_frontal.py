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

# --- MediaPipe setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),          # Nose tip (4)
    (0.0, -330.0, -65.0),     # Chin (152)
    (-225.0, 170.0, -135.0),  # Left eye (33)
    (225.0, 170.0, -135.0),   # Right eye (263)
    (-150.0, -150.0, -125.0), # Mouth left (61)
    (150.0, -150.0, -125.0)   # Mouth right (291)
], dtype=np.float64) / 4.5    # Scale to metric

LANDMARK_IDX = {
    "nose_tip": 4,
    "chin": 152,
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
        [0, h, h / 2],
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

class HeadlessCapture:
    def __init__(self):
        self.running = False
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video device")

    def start(self):
        self.running = True
        print("Starting capture... Press ESC to stop")
        self.capture_loop()

    def stop(self):
        self.running = False
        self.cap.release()
        print("\nCapture stopped")

    def capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            fm_res = face_mesh.process(rgb)

            if fm_res.multi_face_landmarks:
                try:
                    lm = fm_res.multi_face_landmarks[0].landmark

                    head_pose = estimate_head_pose(lm, (h, w))
                    if head_pose:
                        yaw, pitch = head_pose
                        left_iris = lm[468]
                        right_iris = lm[473]
                        left_off = abs(left_iris.x - lm[LANDMARK_IDX["left_eye"]].x) + abs(left_iris.y - lm[LANDMARK_IDX["left_eye"]].y)
                        right_off = abs(right_iris.x - lm[LANDMARK_IDX["right_eye"]].x) + abs(right_iris.y - lm[LANDMARK_IDX["right_eye"]].y)

                        if abs(yaw) < MAX_YAW_DEG and abs(pitch) < MAX_PITCH_DEG and left_off < 0.03 and right_off < 0.03:
                            out = remove_bg_to_white(frame)
                            fname = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".png"
                            cv2.imwrite(os.path.join(OUTPUT_DIR, fname), out)
                            print(f"Captured: {fname}")
                            cv2.waitKey(1000)
                except IndexError:
                    pass

            # Check for ESC key press
            if cv2.waitKey(1) == 27:
                self.stop()
                break

if __name__ == "__main__":
    try:
        capture = HeadlessCapture()
        capture.start()
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if 'capture' in locals():
            capture.stop()
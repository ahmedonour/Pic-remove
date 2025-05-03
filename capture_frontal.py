import cv2
import numpy as np
import mediapipe as mp
from rembg import remove
from datetime import datetime
import os

# --- Configuration ---
# How “strict” to be about frontal:
MAX_YAW_DEG   = 10   # left/right rotation
MAX_PITCH_DEG = 10   # up/down rotation

# Where to save captures
OUTPUT_DIR = "captures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- MediaPipe setup ---
mp_face_mesh = mp.solutions.face_mesh
mp_iris      = mp.solutions.iris
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5)
iris = mp_iris.Iris(static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5)

# 3D model points of facial landmarks for pose estimation
# (nose tip, chin, left eye corner, right eye corner, mouth left, mouth right)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),      # Nose tip
    (0.0, -63.6, -12.5),  # Chin
    (-43.3, 32.7, -26.0), # Left eye left corner
    (43.3, 32.7, -26.0),  # Right eye right corner
    (-28.9, -28.9, -24.1),# Mouth left
    (28.9, -28.9, -24.1), # Mouth right
], dtype=np.float64)

# Indices in MediaPipe FaceMesh for above points
LANDMARK_IDX = {
    "nose_tip":     1,
    "chin":         199,
    "left_eye":     33,
    "right_eye":    263,
    "mouth_left":   61,
    "mouth_right":  291
}

def estimate_head_pose(landmarks, image_shape):
    img_h, img_w = image_shape
    # 2D image points
    image_points = np.array([
        (landmarks[LANDMARK_IDX["nose_tip"]].x * img_w,
         landmarks[LANDMARK_IDX["nose_tip"]].y * img_h),
        (landmarks[LANDMARK_IDX["chin"]].x * img_w,
         landmarks[LANDMARK_IDX["chin"]].y * img_h),
        (landmarks[LANDMARK_IDX["left_eye"]].x * img_w,
         landmarks[LANDMARK_IDX["left_eye"]].y * img_h),
        (landmarks[LANDMARK_IDX["right_eye"]].x * img_w,
         landmarks[LANDMARK_IDX["right_eye"]].y * img_h),
        (landmarks[LANDMARK_IDX["mouth_left"]].x * img_w,
         landmarks[LANDMARK_IDX["mouth_left"]].y * img_h),
        (landmarks[LANDMARK_IDX["mouth_right"]].x * img_w,
         landmarks[LANDMARK_IDX["mouth_right"]].y * img_h),
    ], dtype=np.float64)

    # Camera internals
    focal_length = img_w
    center = (img_w/2, img_h/2)
    cam_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4,1))  # assume no lens distortion

    # Solve PnP
    success, rotation_vec, translation_vec = cv2.solvePnP(
        MODEL_POINTS_3D, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    # Convert to Euler angles
    rot_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rot_mat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    yaw, pitch, roll = euler_angles.flatten()
    return yaw, pitch, roll

def remove_bg_to_white(frame_rgb):
    # send to rembg, get RGBA back
    result = remove(frame_rgb)
    # result is RGBA, convert white background
    rgb = cv2.cvtColor(result, cv2.COLOR_RGBA2RGB)
    # any transparent pixels become black—replace by white
    mask = (result[:,:,3] == 0)
    rgb[mask] = (255,255,255)
    return rgb

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = frame.shape

        # Face mesh detection
        fm_results = face_mesh.process(img_rgb)
        ir_results = iris.process(img_rgb)

        if fm_results.multi_face_landmarks and ir_results.multi_face_landmarks:
            landmarks = fm_results.multi_face_landmarks[0].landmark
            gaze_landmarks = ir_results.multi_face_landmarks[0].landmark

            head_pose = estimate_head_pose(landmarks, (img_h, img_w))
            if head_pose:
                yaw, pitch, roll = head_pose

                # Eye‐gaze: check iris roughly centered in eye socket
                # (landmark 468 is left iris center; 473 is right)
                # we simply check both irises lie near eye center
                # more robust methods exist, but this suffices as a demo
                left_iris = gaze_landmarks[468]
                right_iris = gaze_landmarks[473]
                # normalized device coords: near 0.5,0.5 means centered
                left_off = abs(left_iris.x - landmarks[33].x) + abs(left_iris.y - landmarks[33].y)
                right_off = abs(right_iris.x - landmarks[263].x) + abs(right_iris.y - landmarks[263].y)

                if (abs(yaw) < MAX_YAW_DEG and abs(pitch) < MAX_PITCH_DEG
                    and left_off < 0.03 and right_off < 0.03):
                    # Good frontal + gaze → capture
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    fname = f"{timestamp}.png"
                    # Remove bg & make white
                    out = remove_bg_to_white(frame)
                    cv2.imwrite(os.path.join(OUTPUT_DIR, fname), out[:,:,::-1])
                    print(f"Captured: {fname}")

        # show live (optional)
        cv2.imshow("Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

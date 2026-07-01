import cv2
import mediapipe as mp
import numpy as np
import torch
import pickle
import json
import time
from collections import deque
from pathlib import Path

from feature_extraction import extract_features_from_row, WINDOW_SIZE, N_FEATURES
from model import BiLSTMClassifier

# ── Load model + scaler ──────────────────────────────────────────────────────
CHECKPOINT_DIR = Path("checkpoints")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CHECKPOINT_DIR / "class_names.json") as f:
    CLASS_NAMES = json.load(f)

with open(CHECKPOINT_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

model = BiLSTMClassifier(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device))
model.eval().to(device)

# ── BlazePose setup ──────────────────────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult  = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode     = mp.tasks.vision.RunningMode

LANDMARK_NAMES = [
    "nose","left_eye_inner","left_eye","left_eye_outer",
    "right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_pinky","right_pinky",
    "left_index","right_index","left_thumb","right_thumb",
    "left_hip","right_hip","left_knee","right_knee",
    "left_ankle","right_ankle","left_heel","right_heel",
    "left_foot_index","right_foot_index"
]

CONNECTIONS = {
    "torso":     [(11,12),(11,23),(12,24),(23,24)],
    "left_arm":  [(11,13),(13,15)],
    "right_arm": [(12,14),(14,16)],
    "left_leg":  [(23,25),(25,27),(27,31)],
    "right_leg": [(24,26),(26,28),(28,32)],
}
COLORS = {
    "torso":(255,255,0), "left_arm":(255,0,0),
    "right_arm":(0,0,255), "left_leg":(255,128,0), "right_leg":(0,200,255),
}

latest_result = None
frame_buffer  = deque(maxlen=WINDOW_SIZE)  # rolling 30-frame window
current_pred  = "Waiting..."
current_conf  = 0.0

def on_result(result: PoseLandmarkerResult, output_image, timestamp_ms):
    global latest_result
    latest_result = result

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_full.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=on_result,
    num_poses=1,
)


def landmarks_to_row(landmarks):
    """Convert mediapipe landmark list to a dict matching CSV column names."""
    row = {}
    for i, name in enumerate(LANDMARK_NAMES):
        lm = landmarks[i]
        row[f"{name}_x"] = lm.x
        row[f"{name}_y"] = lm.y
        row[f"{name}_z"] = lm.z
        row[f"{name}_visibility"] = lm.visibility
    return row


def predict_window(window):
    """window: list of 30 feature vectors (each length 78)."""
    X = np.stack(window).reshape(1, WINDOW_SIZE, N_FEATURES)  # (1, 30, 78)
    X_flat = X.reshape(WINDOW_SIZE, N_FEATURES)
    X_flat = scaler.transform(X_flat)
    X = torch.tensor(X_flat.reshape(1, WINDOW_SIZE, N_FEATURES), dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = probs.argmax()
    return CLASS_NAMES[pred_idx], probs[pred_idx]


def draw_skeleton(frame, result):
    if not result or not result.pose_landmarks:
        return frame
    h, w = frame.shape[:2]
    for pose in result.pose_landmarks:
        for part, conns in CONNECTIONS.items():
            for a, b in conns:
                la, lb = pose[a], pose[b]
                if la.visibility > 0.5 and lb.visibility > 0.5:
                    cv2.line(frame,
                             (int(la.x*w), int(la.y*h)),
                             (int(lb.x*w), int(lb.y*h)),
                             COLORS[part], 2)
        for lm in pose:
            if lm.visibility > 0.5:
                cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 5, (0,255,0), -1)
    return frame


cap = cv2.VideoCapture(0)
print("Running inference — press Q to quit")

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        # Buffer features
        if latest_result and latest_result.pose_world_landmarks:
            row      = landmarks_to_row(latest_result.pose_world_landmarks[0])
            features = extract_features_from_row(row)
            frame_buffer.append(features)

            if len(frame_buffer) == WINDOW_SIZE:
                current_pred, current_conf = predict_window(list(frame_buffer))

        frame = draw_skeleton(frame, latest_result)

        # Overlay prediction
        cv2.rectangle(frame, (0, 0), (350, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"{current_pred} ({current_conf*100:.1f}%)",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        cv2.imshow("Motion Classifier", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

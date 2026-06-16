import cv2
import mediapipe as mp
import time

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Skeleton connections grouped by body part
CONNECTIONS = {
    "torso":  [(11,12),(11,23),(12,24),(23,24)],
    "left_arm":  [(11,13),(13,15)],
    "right_arm": [(12,14),(14,16)],
    "left_leg":  [(23,25),(25,27),(27,31)],
    "right_leg": [(24,26),(26,28),(28,32)],
}

COLORS = {
    "torso":     (255, 255, 0),   # yellow
    "left_arm":  (255, 0, 0),     # blue
    "right_arm": (0, 0, 255),     # red
    "left_leg":  (255, 128, 0),   # orange
    "right_leg": (0, 200, 255),   # gold
}

def draw_skeleton(frame, result):
    if not result or not result.pose_landmarks:
        return frame

    h, w = frame.shape[:2]

    for pose in result.pose_landmarks:
        # Draw connections
        for part, connections in CONNECTIONS.items():
            color = COLORS[part]
            for a, b in connections:
                lm_a, lm_b = pose[a], pose[b]
                if lm_a.visibility > 0.5 and lm_b.visibility > 0.5:
                    x1, y1 = int(lm_a.x * w), int(lm_a.y * h)
                    x2, y2 = int(lm_b.x * w), int(lm_b.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), color, 2)

        # Draw joints
        for lm in pose:
            if lm.visibility > 0.5:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    return frame

latest_result = None

def on_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_full.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=on_result,
    # num_poses=1,
    num_poses=10,
)

cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        frame = draw_skeleton(frame, latest_result)

        # if latest_result and latest_result.pose_world_landmarks:
        #     # Print hip-center-relative 3D coords for left wrist (landmark 15)
        #     lm = latest_result.pose_world_landmarks[0][15]
        #     print(f"Left wrist 3D: ({lm.x:.3f}, {lm.y:.3f}, {lm.z:.3f})")

        cv2.imshow("BlazePose", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

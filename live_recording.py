import cv2
import mediapipe as mp
import csv
import os
import sys
import time

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH  = "pose_landmarker_full.task"
MAX_PEOPLE  = 10
# ─────────────────────────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    print("Usage: python live_recording.py <name>")
    sys.exit(1)

name = sys.argv[1]
os.makedirs("recorded_data", exist_ok=True)
OUTPUT_CSV   = f"recorded_data/{name}.csv"
OUTPUT_VIDEO = f"recorded_data/{name}.mp4"

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]
NUM_LANDMARKS = len(LANDMARK_NAMES)  # 33

CONNECTIONS = {
    "torso":     [(11, 12), (11, 23), (12, 24), (23, 24)],
    "left_arm":  [(11, 13), (13, 15)],
    "right_arm": [(12, 14), (14, 16)],
    "left_leg":  [(23, 25), (25, 27), (27, 31)],
    "right_leg": [(24, 26), (26, 28), (28, 32)],
}

COLORS = {
    "torso":     (255, 255, 0),
    "left_arm":  (255, 0, 0),
    "right_arm": (0, 0, 255),
    "left_leg":  (255, 128, 0),
    "right_leg": (0, 200, 255),
}


def build_csv_header():
    header = ["frame", "person_id"]
    for lm_name in LANDMARK_NAMES:
        header += [f"{lm_name}_x", f"{lm_name}_y", f"{lm_name}_z", f"{lm_name}_visibility"]
    for lm_name in LANDMARK_NAMES:
        header += [f"{lm_name}_world_x", f"{lm_name}_world_y", f"{lm_name}_world_z"]
    return header


EXPECTED_COLS = len(build_csv_header())


def build_row(frame_idx, person_id, image_pose, world_pose):
    row = [frame_idx, person_id]

    for lm in image_pose:
        row += [lm.x, lm.y, lm.z, lm.visibility]

    if world_pose is not None and len(world_pose) == NUM_LANDMARKS:
        for lm in world_pose:
            row += [lm.x, lm.y, lm.z]
    else:
        row += [""] * (NUM_LANDMARKS * 3)

    return row


def draw_skeleton(frame, result):
    if not result or not result.pose_landmarks:
        return frame
    h, w = frame.shape[:2]
    for pose in result.pose_landmarks:
        for part, connections in CONNECTIONS.items():
            color = COLORS[part]
            for a, b in connections:
                lm_a, lm_b = pose[a], pose[b]
                if lm_a.visibility > 0.5 and lm_b.visibility > 0.5:
                    x1, y1 = int(lm_a.x * w), int(lm_a.y * h)
                    x2, y2 = int(lm_b.x * w), int(lm_b.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        for lm in pose:
            if lm.visibility > 0.5:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    return frame


latest_result = None


def on_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result


BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=on_result,
    num_poses=MAX_PEOPLE,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
)

cap = cv2.VideoCapture(0)
fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

print(f"Recording to recorded_data/{name}.(csv|mp4) — press Q to stop")
print(f"Expecting {EXPECTED_COLS} columns per row "
      f"({NUM_LANDMARKS} landmarks x,y,z,visibility + {NUM_LANDMARKS} landmarks world x,y,z)")

total_frames = 0
frames_with_pose = 0
frames_missing_world = 0
rows_written = 0

with open(OUTPUT_CSV, "w", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(build_csv_header())

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            landmarker.detect_async(mp_image, int(time.time() * 1000))

            frame = draw_skeleton(frame, latest_result)

            if latest_result and latest_result.pose_landmarks:
                frames_with_pose += 1
                world_landmarks_list = latest_result.pose_world_landmarks or []
                for person_id, image_pose in enumerate(latest_result.pose_landmarks):
                    world_pose = (
                        world_landmarks_list[person_id]
                        if person_id < len(world_landmarks_list)
                        else None
                    )
                    if world_pose is None:
                        frames_missing_world += 1

                    row = build_row(frame_idx, person_id, image_pose, world_pose)
                    assert len(row) == EXPECTED_COLS, (
                        f"Row has {len(row)} columns, expected {EXPECTED_COLS}"
                    )
                    csv_writer.writerow(row)
                    rows_written += 1

            writer.write(frame)
            cv2.imshow("BlazePose — Recording", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1

cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"Saved: {OUTPUT_CSV}")
print(f"Saved: {OUTPUT_VIDEO}")
print("--- Recording summary ---")
print(f"Camera frames captured:      {total_frames}")
print(f"Frames with a detected pose: {frames_with_pose}")
print(f"Frames with no pose at all:  {total_frames - frames_with_pose}")
print(f"Rows written to CSV:         {rows_written}")
print(f"Rows missing world coords:   {frames_missing_world}")

import numpy as np
import pandas as pd
from pathlib import Path
import json

# ── Landmark indices (BlazePose 33 keypoints) ────────────────────────────────
LM = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13,    "right_elbow": 14,
    "left_wrist": 15,    "right_wrist": 16,
    "left_hip": 23,      "right_hip": 24,
    "left_knee": 25,     "right_knee": 26,
    "left_ankle": 27,    "right_ankle": 28,
    "left_heel": 29,     "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
    "left_pinky": 17,    "right_pinky": 18,
    "left_index": 19,    "right_index": 20,
    "left_thumb": 21,    "right_thumb": 22,
}

# 12 joint angle triplets from the paper
ANGLE_TRIPLETS = [
    ("left_hip",      "left_shoulder",  "left_elbow"),
    ("right_hip",     "right_shoulder", "right_elbow"),
    ("left_shoulder", "left_elbow",     "left_wrist"),
    ("right_shoulder","right_elbow",    "right_wrist"),
    ("left_hip",      "left_knee",      "left_ankle"),
    ("right_hip",     "right_knee",     "right_ankle"),
    ("left_shoulder", "left_hip",       "left_knee"),
    ("right_shoulder","right_hip",      "right_knee"),
    ("left_knee",     "left_ankle",     "left_heel"),
    ("right_knee",    "right_ankle",    "right_heel"),
    ("left_ankle",    "left_heel",      "left_foot_index"),
    ("right_ankle",   "right_heel",     "right_foot_index"),
]

# 22 coordinate landmarks from the paper (x, y, z each)
COORD_LANDMARKS = [
    "left_shoulder", "right_shoulder",
    "left_hip",      "right_hip",
    "left_knee",     "right_knee",
    "left_elbow",    "right_elbow",
    "left_wrist",    "right_wrist",
    "left_ankle",    "right_ankle",
    "left_heel",     "right_heel",
    "left_pinky",    "right_pinky",
    "left_index",    "right_index",
    "left_thumb",    "right_thumb",
    "left_foot_index", "right_foot_index",
]

WINDOW_SIZE = 30  # frames per sequence
# 12 angles + 22 landmarks * 3 coords = 12 + 66 = 78 features
N_FEATURES = 78


def get_xyz(row, name):
    return np.array([row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_z"]], dtype=np.float32)


def compute_angle(a, b, c):
    """Angle at joint b formed by a-b-c, in degrees."""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def extract_features_from_row(row):
    """Extract 78 features (12 angles + 66 coords) from a single frame row."""
    features = []

    # 12 angles
    for a_name, b_name, c_name in ANGLE_TRIPLETS:
        a = get_xyz(row, a_name)
        b = get_xyz(row, b_name)
        c = get_xyz(row, c_name)
        features.append(compute_angle(a, b, c))

    # 66 coordinates (22 landmarks * 3)
    for name in COORD_LANDMARKS:
        xyz = get_xyz(row, name)
        features.extend(xyz.tolist())

    return np.array(features, dtype=np.float32)


def load_csv_to_windows(csv_path, label, window_size=WINDOW_SIZE):
    """
    Load a recorded CSV and return (windows, labels).
    Each window is shape (window_size, N_FEATURES).
    Assumes person_id == 0 (single person) — adjust if needed.
    """
    df = pd.read_csv(csv_path)

    # Filter to person 0 and sort by frame
    df = df[df["person_id"] == 0].sort_values("frame").reset_index(drop=True)

    # Rename columns: CSV has e.g. "left_shoulder_x" which matches our naming
    features_per_frame = []
    for _, row in df.iterrows():
        features_per_frame.append(extract_features_from_row(row))

    # Slide a window of 30 frames (no overlap for training efficiency)
    windows, labels = [], []
    for i in range(0, len(features_per_frame) - window_size + 1, window_size):
        window = np.stack(features_per_frame[i:i + window_size])  # (30, 78)
        windows.append(window)
        labels.append(label)

    return windows, labels


def build_dataset(data_dir, class_names, window_size=WINDOW_SIZE):
    """
    data_dir/
      class_name_1/  <- folder of CSVs for that class
        recording1.csv
        recording2.csv
      class_name_2/
        ...

    Returns X: (N, 30, 78), y: (N,) as numpy arrays.
    """
    all_windows, all_labels = [], []

    for label_idx, class_name in enumerate(class_names):
        class_dir = Path(data_dir) / class_name
        csv_files = list(class_dir.glob("*.csv"))
        print(f"  {class_name}: {len(csv_files)} recordings")

        for csv_path in csv_files:
            windows, labels = load_csv_to_windows(csv_path, label_idx, window_size)
            all_windows.extend(windows)
            all_labels.extend(labels)

    X = np.stack(all_windows)   # (N, 30, 78)
    y = np.array(all_labels)    # (N,)
    print(f"\nDataset: {X.shape[0]} windows, {X.shape[1]} timesteps, {X.shape[2]} features")
    return X, y

import numpy as np
import pandas as pd
from pathlib import Path

# ── ZED BODY_38 format ────────────────────────────────────────────────────
# Columns in the CSV: frame,timestamp_s,body_id,tracking_state,confidence,
#                      joint_00_x,joint_00_y,joint_00_z, ..., joint_37_z
#
# We don't hardcode per-joint semantics (e.g. "which index is the left wrist")
# because that mapping wasn't confidently verifiable for BODY_38 at the time
# this was written -- getting it wrong would silently corrupt every angle
# feature. Instead we use all 38 raw joint coordinates, normalized to be
# translation- and scale-invariant. This sidesteps the need for a joint name
# map entirely while still giving the BiLSTM full pose information.
#
# One assumption: joint index 0 (PELVIS) is the root joint, which is the
# standard convention for ZED's BODY_34/BODY_38 skeletons. Verify against
# your installed SDK if needed:
#   import pyzed.sl as sl; print(list(sl.BODY_38_PARTS))

N_JOINTS = 38
ROOT_JOINT_INDEX = 0  # assumed PELVIS -- see note above
N_FEATURES = N_JOINTS * 3  # 114: normalized (x, y, z) for every joint

WINDOW_SIZE = 30    # frames per training sequence -- tune to match your ZED capture FPS
WINDOW_STRIDE = 15  # 50% overlap between windows; set equal to WINDOW_SIZE for no overlap

TRACKING_STATE_OK = "OK"   # from sl.OBJECT_TRACKING_STATE (OFF/OK/SEARCHING/TERMINATE)
MIN_CONFIDENCE = 0.0       # 0-100; raise to drop low-confidence detections, 0 = no filtering

JOINT_COLS = [f"joint_{i:02d}_{axis}" for i in range(N_JOINTS) for axis in ("x", "y", "z")]


def _joint_matrix(row):
    """Return an (N_JOINTS, 3) array of joint coordinates for a single frame row."""
    vals = row[JOINT_COLS].to_numpy(dtype=np.float32)
    return vals.reshape(N_JOINTS, 3)


def normalize_pose(joints):
    """
    Make a single frame's pose translation- and scale-invariant.
      - Translation: subtract the root joint (assumed pelvis, index 0).
      - Scale: divide by the mean distance from the root to all other
        joints, a rough proxy for body size / distance from the camera.
    """
    root = joints[ROOT_JOINT_INDEX]
    centered = joints - root
    scale = np.linalg.norm(centered, axis=1).mean()
    if scale < 1e-6:
        scale = 1.0
    return centered / scale


def extract_features_from_row(row):
    """Extract N_FEATURES (114) normalized coordinates from a single frame row."""
    joints = _joint_matrix(row)
    normalized = normalize_pose(joints)
    return normalized.reshape(-1).astype(np.float32)


def load_csv_to_windows(csv_path, label, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    """
    Load one recorded ZED CSV and return (windows, labels).

    Handles multiple tracked people (body_id) in the same recording:
    each body_id's track is filtered and windowed independently, since a
    single frame index can contain several people's rows.
    """
    df = pd.read_csv(csv_path)

    windows, labels = [], []

    for body_id, track in df.groupby("body_id"):
        track = track.sort_values("frame").reset_index(drop=True)

        # Keep only well-tracked, sufficiently confident frames
        if "tracking_state" in track.columns:
            track = track[track["tracking_state"].astype(str).str.upper() == TRACKING_STATE_OK]
        if "confidence" in track.columns and MIN_CONFIDENCE > 0:
            track = track[track["confidence"] >= MIN_CONFIDENCE]
        track = track.reset_index(drop=True)

        if len(track) < window_size:
            continue

        features_per_frame = [extract_features_from_row(row) for _, row in track.iterrows()]

        for i in range(0, len(features_per_frame) - window_size + 1, stride):
            window = np.stack(features_per_frame[i:i + window_size])  # (window_size, N_FEATURES)
            windows.append(window)
            labels.append(label)

    return windows, labels


def build_dataset(data_dir, class_names, window_size=WINDOW_SIZE, stride=WINDOW_STRIDE):
    """
    data_dir/
      class_name_1/  <- folder of CSVs for that class
        recording1.csv
      class_name_2/
        ...

    Returns X: (N, window_size, N_FEATURES), y: (N,) as numpy arrays.
    """
    all_windows, all_labels = [], []

    for label_idx, class_name in enumerate(class_names):
        class_dir = Path(data_dir) / class_name
        csv_files = list(class_dir.glob("*.csv"))
        print(f"  {class_name}: {len(csv_files)} recordings")

        for csv_path in csv_files:
            windows, labels = load_csv_to_windows(csv_path, label_idx, window_size, stride)
            all_windows.extend(windows)
            all_labels.extend(labels)

    X = np.stack(all_windows)   # (N, window_size, N_FEATURES)
    y = np.array(all_labels)    # (N,)
    print(f"\nDataset: {X.shape[0]} windows, {X.shape[1]} timesteps, {X.shape[2]} features")
    return X, y

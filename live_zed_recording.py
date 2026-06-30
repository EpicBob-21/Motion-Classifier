"""
live_zed_recording.py
---------------------
Records multi-person ZED BODY_38 landmarks to CSV and raw video to MP4.

Usage:
    python live_zed_recording.py <recording_name>

Outputs:
    <recording_name>_landmarks.csv   — per-frame, per-body joint positions
    <recording_name>_video.mp4       — raw left-camera video

CSV columns:
    frame, timestamp_s, body_id, tracking_state, confidence,
    joint_00_x, joint_00_y, joint_00_z, ..., joint_37_x, joint_37_y, joint_37_z
    (38 joints × 3 coords = 114 position columns)
"""

import sys
import csv
import time
import signal
import argparse

import numpy as np
import cv2
import pyzed.sl as sl
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# SMPL mapping & normalization (kept for optional use)
# ---------------------------------------------------------------------------

class ZEDtoSMPLMapper:
    """
    Converts ZED BODY_38 format to SMPL format with AIST++ normalization.
    Call extract_smpl_from_body() to get normalized SMPL data, or
    smpl_to_aist_format() to get a (1, 219) AIST++ motion vector.
    """

    ZED_TO_SMPL_MAP = {
        0:  (0,  "Pelvis"),
        1:  (18, "Left Hip"),
        2:  (19, "Right Hip"),
        3:  (1,  "Spine1"),
        4:  (20, "Left Knee"),
        5:  (21, "Right Knee"),
        6:  (2,  "Spine2"),
        7:  (22, "Left Ankle"),
        8:  (23, "Right Ankle"),
        9:  (3,  "Spine3"),
        10: (24, "Left Foot"),
        11: (25, "Right Foot"),
        12: (4,  "Neck"),
        13: (10, "Left Clavicle"),
        14: (11, "Right Clavicle"),
        15: (5,  "Head/Nose"),
        16: (12, "Left Shoulder"),
        17: (13, "Right Shoulder"),
        18: (14, "Left Elbow"),
        19: (15, "Right Elbow"),
        20: (16, "Left Wrist"),
        21: (17, "Right Wrist"),
        22: (30, "Left Hand"),
        23: (31, "Right Hand"),
    }

    def __init__(self, reference_height=1.7):
        self.smpl_joint_count = 24
        self.zed_joint_count = 38
        self.reference_height = reference_height

    def zed_to_smpl_positions(self, zed_keypoints):
        if zed_keypoints.shape[0] != 38:
            raise ValueError(f"Expected 38 ZED joints, got {zed_keypoints.shape[0]}")
        smpl_positions = np.zeros((24, 3))
        for smpl_idx, (zed_idx, _) in self.ZED_TO_SMPL_MAP.items():
            smpl_positions[smpl_idx] = zed_keypoints[zed_idx]
        return smpl_positions

    def quaternion_to_axis_angle(self, quat):
        if np.linalg.norm(quat) < 1e-6:
            return np.zeros(3)
        rot = R.from_quat(quat / np.linalg.norm(quat))
        return rot.as_rotvec()

    def zed_to_smpl_pose(self, zed_quaternions):
        if zed_quaternions.shape[0] != 38:
            raise ValueError(f"Expected 38 ZED quaternions, got {zed_quaternions.shape[0]}")
        smpl_pose = np.zeros((24, 3))
        for smpl_idx, (zed_idx, _) in self.ZED_TO_SMPL_MAP.items():
            quat = zed_quaternions[zed_idx]
            smpl_pose[smpl_idx] = self.quaternion_to_axis_angle(quat)
        return smpl_pose.flatten()

    def compute_body_height(self, positions):
        pelvis = positions[0]
        head   = positions[15]
        return np.abs(head[1] - pelvis[1])

    def compute_scaling_factor(self, positions):
        h = self.compute_body_height(positions)
        return self.reference_height / h if h > 0.1 else 1.0

    def compute_facing_direction(self, positions):
        left_shoulder  = positions[16]
        right_shoulder = positions[17]
        shoulder_vec = left_shoulder - right_shoulder
        shoulder_vec_xz = np.array([shoulder_vec[0], 0, shoulder_vec[2]])
        norm = np.linalg.norm(shoulder_vec_xz)
        if not np.isfinite(norm) or norm < 0.01:
            return 0.0
        shoulder_vec_xz /= norm
        forward_vec = np.cross(shoulder_vec_xz, np.array([0, 1, 0]))
        reference = np.array([0, 0, 1])
        cos_angle = np.dot(forward_vec, reference)
        sin_angle = np.cross(forward_vec, reference)[1]
        return np.arctan2(sin_angle, cos_angle)

    def rotate_around_y_axis(self, positions, angle):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R_y = np.array([[cos_a, 0, sin_a], [0, 1, 0], [-sin_a, 0, cos_a]])
        return positions @ R_y.T

    def normalize_pose(self, positions, pose, global_quat):
        pelvis_pos         = positions[0].copy()
        centered_positions = positions - pelvis_pos
        yaw                = self.compute_facing_direction(centered_positions)
        normalized_positions = self.rotate_around_y_axis(centered_positions, -yaw)
        scaling            = self.compute_scaling_factor(positions)
        scaled_positions   = normalized_positions * scaling

        if not np.isfinite(yaw):
            yaw = 0.0
        yaw_correction = R.from_euler('y', -yaw)
        norm = np.linalg.norm(global_quat)
        safe_quat = global_quat / norm if norm > 1e-6 else np.array([0, 0, 0, 1.0])
        corrected_global_quat = (yaw_correction * R.from_quat(safe_quat)).as_quat()

        return {
            'positions':          scaled_positions,
            'pose':               pose,
            'translation':        pelvis_pos,
            'scaling':            scaling,
            'global_orientation': corrected_global_quat,
            'yaw_correction':     yaw,
        }

    def extract_smpl_from_body(self, body, normalize=True):
        zed_keypoints  = np.array(body.keypoint)
        zed_quaternions = np.array([
            body.local_orientation_per_joint[i] for i in range(38)
        ])
        global_quat   = np.array(body.global_root_orientation)
        smpl_positions = self.zed_to_smpl_positions(zed_keypoints)
        smpl_pose      = self.zed_to_smpl_pose(zed_quaternions)
        if normalize:
            return self.normalize_pose(smpl_positions, smpl_pose, global_quat)
        return {'positions': smpl_positions, 'pose': smpl_pose, 'global_orientation': global_quat}

    def smpl_to_aist_format(self, smpl_data):
        pose        = smpl_data['pose']
        translation = smpl_data['translation']
        rotation_matrices = []
        for j in range(24):
            aa = pose[j * 3:(j + 1) * 3]
            rot_matrix = np.eye(3) if np.allclose(aa, 0) else R.from_rotvec(aa).as_matrix()
            rotation_matrices.append(rot_matrix.flatten(order='F'))
        rotation_part = np.concatenate(rotation_matrices)
        aist_vector   = np.concatenate([translation, rotation_part])
        return aist_vector.reshape(1, 219)


# ---------------------------------------------------------------------------
# CSV header builder
# ---------------------------------------------------------------------------

def build_csv_header():
    """Return the full list of CSV column names."""
    meta = ["frame", "timestamp_s", "body_id", "tracking_state", "confidence"]
    joints = []
    for j in range(38):
        joints += [f"joint_{j:02d}_x", f"joint_{j:02d}_y", f"joint_{j:02d}_z"]
    return meta + joints


# ---------------------------------------------------------------------------
# Main recorder
# ---------------------------------------------------------------------------

class ZEDRecorder:
    def __init__(
        self,
        recording_name: str,
        resolution=sl.RESOLUTION.HD720,
        fps: int = 30,
        detection_confidence: int = 40,
        skeleton_smoothing: float = 0.5,
    ):
        self.recording_name = recording_name
        self.target_fps     = fps
        self._running       = False

        # ---- File paths ----
        self.csv_path   = f"zed_data/{recording_name}_landmarks.csv"
        self.video_path = f"zed_data/{recording_name}_video.mp4"

        # ---- ZED init ----
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = resolution
        init_params.camera_fps        = fps
        init_params.depth_mode        = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units  = sl.UNIT.METER

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")

        pos_params = sl.PositionalTrackingParameters()
        err = self.zed.enable_positional_tracking(pos_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to enable positional tracking: {err}")

        body_params = sl.BodyTrackingParameters()
        body_params.body_format       = sl.BODY_FORMAT.BODY_38
        body_params.detection_model   = sl.BODY_TRACKING_MODEL.HUMAN_BODY_MEDIUM
        body_params.enable_tracking   = True
        body_params.enable_body_fitting = True

        err = self.zed.enable_body_tracking(body_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to enable body tracking: {err}")

        self._bt_runtime = sl.BodyTrackingRuntimeParameters()
        self._bt_runtime.detection_confidence_threshold = detection_confidence
        self._bt_runtime.skeleton_smoothing             = skeleton_smoothing

        self._bodies       = sl.Bodies()
        self._runtime      = sl.RuntimeParameters()
        self._image_mat    = sl.Mat()

        # Determine actual frame size from camera
        cam_info   = self.zed.get_camera_information()
        frame_w    = cam_info.camera_configuration.resolution.width
        frame_h    = cam_info.camera_configuration.resolution.height
        self._frame_size = (frame_w, frame_h)

        # ---- OpenCV VideoWriter ----
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self.video_path, fourcc, fps, self._frame_size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {self.video_path}")

        # ---- CSV writer ----
        self._csv_file   = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(build_csv_header())

        print(f"[ZEDRecorder] Camera: {frame_w}×{frame_h} @ {fps} fps")
        print(f"[ZEDRecorder] Landmarks → {self.csv_path}")
        print(f"[ZEDRecorder] Video     → {self.video_path}")
        print("[ZEDRecorder] Ready ✅  (Ctrl+C to stop)\n")

    # ------------------------------------------------------------------
    def _grab_frame(self):
        """Grab one camera frame; return (bodies, bgr_frame) or ([], None) on failure."""
        if self.zed.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
            return [], None

        self.zed.retrieve_bodies(self._bodies, self._bt_runtime)
        self.zed.retrieve_image(self._image_mat, sl.VIEW.LEFT)

        bodies = self._bodies.body_list
        raw    = self._image_mat.get_data()
        frame  = raw[:, :, :3].copy() if raw is not None else None  # BGRA → BGR

        return bodies, frame

    def _write_csv_rows(self, frame_idx: int, timestamp_s: float, bodies):
        """Write one CSV row per detected body."""
        for body in bodies:
            keypoints = np.array(body.keypoint)  # (38, 3), may contain NaN

            meta = [
                frame_idx,
                f"{timestamp_s:.6f}",
                body.id,
                str(body.tracking_state),
                f"{body.confidence:.2f}",
            ]

            joint_vals = []
            for j in range(38):
                if j < len(keypoints):
                    x, y, z = keypoints[j]
                    joint_vals += [
                        "" if not np.isfinite(x) else f"{x:.6f}",
                        "" if not np.isfinite(y) else f"{y:.6f}",
                        "" if not np.isfinite(z) else f"{z:.6f}",
                    ]
                else:
                    joint_vals += ["", "", ""]

            self._csv_writer.writerow(meta + joint_vals)

    # ------------------------------------------------------------------
    def run(self):
        self._running  = True
        frame_idx      = 0
        start_time     = time.time()
        last_status    = start_time
        status_every_s = 2.0

        while self._running:
            bodies, frame = self._grab_frame()
            timestamp_s   = time.time() - start_time

            # Write video frame
            if frame is not None:
                if frame.shape[1] != self._frame_size[0] or frame.shape[0] != self._frame_size[1]:
                    frame = cv2.resize(frame, self._frame_size)
                self._writer.write(frame)

            # Write CSV rows (one per body)
            if bodies:
                self._write_csv_rows(frame_idx, timestamp_s, bodies)

            frame_idx += 1

            # Status print
            now = time.time()
            if now - last_status >= status_every_s:
                elapsed   = now - start_time
                actual_fps = frame_idx / elapsed if elapsed > 0 else 0
                print(
                    f"  t={elapsed:6.1f}s | frame={frame_idx:5d} | "
                    f"fps={actual_fps:5.1f} | bodies={len(bodies)}"
                )
                last_status = now

        self._stop()

    def stop(self):
        """Signal the run loop to stop (call from signal handler)."""
        self._running = False

    def _stop(self):
        """Flush and close all resources."""
        print("\n[ZEDRecorder] Stopping — flushing files...")
        self._csv_file.flush()
        self._csv_file.close()
        self._writer.release()
        self.zed.disable_body_tracking()
        self.zed.disable_positional_tracking()
        self.zed.close()
        print(f"[ZEDRecorder] Saved landmarks → {self.csv_path}")
        print(f"[ZEDRecorder] Saved video     → {self.video_path}")
        print("[ZEDRecorder] Done ✅")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Record ZED BODY_38 landmarks (CSV) + raw video (MP4)."
    )
    parser.add_argument(
        "recording_name",
        type=str,
        help="Base name for output files (e.g. 'session_01' → session_01_landmarks.csv + session_01_video.mp4)",
    )
    parser.add_argument("--fps",        type=int,   default=30,   help="Camera FPS (default: 30)")
    parser.add_argument("--resolution", type=str,   default="HD720",
                        choices=["HD2K", "HD1080", "HD720", "VGA"],
                        help="Camera resolution (default: HD720)")
    parser.add_argument("--confidence", type=int,   default=40,   help="Body detection confidence threshold (default: 40)")
    parser.add_argument("--smoothing",  type=float, default=0.5,  help="Skeleton smoothing factor 0–1 (default: 0.5)")
    return parser.parse_args()


RESOLUTION_MAP = {
    "HD2K":  sl.RESOLUTION.HD2K,
    "HD1080": sl.RESOLUTION.HD1080,
    "HD720": sl.RESOLUTION.HD720,
    "VGA":   sl.RESOLUTION.VGA,
}


def main():
    args = parser_args = parse_args()

    recorder = ZEDRecorder(
        recording_name=args.recording_name,
        resolution=RESOLUTION_MAP[args.resolution],
        fps=args.fps,
        detection_confidence=args.confidence,
        skeleton_smoothing=args.smoothing,
    )

    # Graceful Ctrl+C
    def _signal_handler(sig, frame):
        print("\n[INFO] Interrupt received — stopping...")
        recorder.stop()

    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    recorder.run()


if __name__ == "__main__":
    main()

import glob
import math
import os
import time
from collections import Counter
from math import gcd

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly

# ---------------- Config ----------------
SOUND_DIR = "flowedit_sounds"
FLIP = 1               # set to -1 if left/right feel swapped
SMOOTHING = 0.3        # 0..1, lower = smoother but laggier

ROCK_SPEED = 60.0      # deg/s, how fast the lean must swing to count as a rock
ROCK_MIN_ANGLE = 8.0   # deg, minimum sideways lean when the rock fires
BOOMERANG_OUT = 0.6    # seconds to sweep out to the end file
BOOMERANG_BACK = 0.8   # seconds to sweep back to the current body angle
ROCK_COOLDOWN = 0.3    # seconds after a boomerang before another rock registers
# ----------------------------------------

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

CONNECTIONS = {
    "torso":  [(11,12),(11,23),(12,24),(23,24)],
    "left_arm":  [(11,13),(13,15)],
    "right_arm": [(12,14),(14,16)],
    "left_leg":  [(23,25),(25,27),(27,31)],
    "right_leg": [(24,26),(26,28),(28,32)],
}

COLORS = {
    "torso":     (255, 255, 0),
    "left_arm":  (255, 0, 0),
    "right_arm": (0, 0, 255),
    "left_leg":  (255, 128, 0),
    "right_leg": (0, 200, 255),
}


class ScrubPlayer:
    """Loads every FLAC into memory and plays them in sync.
    Changing `target` switches files at the same playback position,
    with a short crossfade to avoid clicks."""

    def __init__(self, folder):
        paths = sorted(glob.glob(os.path.join(folder, "*.flac")))
        if not paths:
            raise FileNotFoundError(f"No .flac files found in {folder}")
        self.names = [os.path.basename(p) for p in paths]

        loaded = [sf.read(p, dtype="float32", always_2d=True) for p in paths]

        # Play everything at the most common sample rate
        self.samplerate = Counter(sr for _, sr in loaded).most_common(1)[0][0]

        tracks = []
        for data, sr in loaded:
            if sr != self.samplerate:
                g = gcd(sr, self.samplerate)
                data = resample_poly(data, self.samplerate // g, sr // g, axis=0).astype(np.float32)
            tracks.append(data)

        self.channels = max(t.shape[1] for t in tracks)
        self.tracks = [
            t if t.shape[1] == self.channels else np.repeat(t[:, :1], self.channels, axis=1)
            for t in tracks
        ]
        self.pos = 0
        self.current = 0
        self.target = 0

    def _chunk(self, idx, n):
        t = self.tracks[idx]
        return t[(self.pos + np.arange(n)) % len(t)]

    def _callback(self, outdata, frames, time_info, status):
        out = self._chunk(self.target, frames)
        if self.target != self.current:
            old = self._chunk(self.current, frames)
            ramp = np.linspace(0, 1, frames, dtype=np.float32)[:, None]
            out = old * (1 - ramp) + out * ramp
            self.current = self.target
        self.pos += frames
        outdata[:] = out

    def stream(self):
        return sd.OutputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            blocksize=1024,
            callback=self._callback,
        )


def torso_lean_deg(world):
    """Sideways lean of the upper body. + = leaning to your left, - = to your right.
    Measured along the shoulder line, so it's independent of torso yaw."""
    l, r = world[11], world[12]
    lh, rh = world[23], world[24]

    # unit vector along the shoulders (horizontal plane)
    ax, az = l.x - r.x, l.z - r.z
    norm = math.hypot(ax, az) or 1e-6
    ax, az = ax / norm, az / norm

    # hips -> shoulders
    vx = (l.x + r.x) / 2 - (lh.x + rh.x) / 2
    vy = (l.y + r.y) / 2 - (lh.y + rh.y) / 2
    vz = (l.z + r.z) / 2 - (lh.z + rh.z) / 2

    lateral = vx * ax + vz * az
    vertical = -vy  # MediaPipe world y points down
    return math.degrees(math.atan2(lateral, vertical))


class RockDetector:
    """Flags a rock when the upper body leans sideways quickly.
    Returns 'left', 'right' or None."""

    def __init__(self):
        self.prev = None      # (timestamp, lean)
        self.speed = 0.0      # signed deg/s, for the HUD
        self.lean = 0.0

    def update(self, world, ts_s):
        if min(world[i].visibility for i in (11, 12, 23, 24)) < 0.5:
            self.prev = None
            return None

        lean = torso_lean_deg(world)
        rock = None
        if self.prev is not None:
            dt = ts_s - self.prev[0]
            if 0 < dt < 0.2:
                self.speed = (lean - self.prev[1]) / dt
                # must be leaning *out* (speed and lean same sign), past the min angle
                if (abs(self.speed) > ROCK_SPEED and abs(lean) > ROCK_MIN_ANGLE
                        and self.speed * lean > 0):
                    rock = "left" if lean > 0 else "right"
        self.lean = lean
        self.prev = (ts_s, lean)
        return rock


def smoothstep(p):
    p = min(max(p, 0.0), 1.0)
    return p * p * (3 - 2 * p)


def torso_yaw_deg(world_landmarks):
    """0 = facing camera, +90 = turned to your left, -90 = turned to your right."""
    l, r = world_landmarks[11], world_landmarks[12]
    return math.degrees(math.atan2(l.z - r.z, l.x - r.x))


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
                    cv2.line(frame, (int(lm_a.x * w), int(lm_a.y * h)),
                             (int(lm_b.x * w), int(lm_b.y * h)), color, 2)
        for lm in pose:
            if lm.visibility > 0.5:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 5, (0, 255, 0), -1)
    return frame


latest_result = None
latest_ts = 0

def on_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result, latest_ts
    latest_result = result
    latest_ts = timestamp_ms


options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_full.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=on_result,
    num_poses=1,  # one person controls the audio
)

player = ScrubPlayer(SOUND_DIR)
n_files = len(player.names)
rocks = RockDetector()

yaw_smooth = None
body_idx = 0.0       # file index implied by torso angle (continuous)
play_idx = 0.0       # file index actually playing (continuous)
boomerang = None     # active boomerang state, or None
boomerang_end = 0.0  # when the last boomerang finished
last_ts = None

cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as landmarker, player.stream():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        frame = draw_skeleton(frame, latest_result)
        now = time.time()

        if latest_result and latest_result.pose_world_landmarks:
            world = latest_result.pose_world_landmarks[0]

            # Torso angle -> file index (+90 left -> 00_input, -90 right -> final)
            yaw = torso_yaw_deg(world)
            yaw_smooth = yaw if yaw_smooth is None else SMOOTHING * yaw + (1 - SMOOTHING) * yaw_smooth
            frac = float(np.clip((90 - FLIP * yaw_smooth) / 180, 0, 1))
            body_idx = frac * (n_files - 1)

            # Rock detection (once per new pose result)
            if latest_ts != last_ts:
                last_ts = latest_ts
                rock = rocks.update(world, latest_ts / 1000)
                if rock and boomerang is None and now - boomerang_end > ROCK_COOLDOWN:
                    toward_first = (rock == "left") == (FLIP == 1)
                    boomerang = {
                        "phase": "out",
                        "t0": now,
                        "start": play_idx,
                        "end": 0 if toward_first else n_files - 1,
                    }

        # Decide which file index should be playing
        if boomerang is None:
            play_idx = body_idx
        elif boomerang["phase"] == "out":
            p = (now - boomerang["t0"]) / BOOMERANG_OUT
            play_idx = boomerang["start"] + (boomerang["end"] - boomerang["start"]) * smoothstep(p)
            if p >= 1:
                boomerang["phase"], boomerang["t0"] = "back", now
        else:  # coming back to wherever the body is pointing right now
            p = (now - boomerang["t0"]) / BOOMERANG_BACK
            play_idx = boomerang["end"] + (body_idx - boomerang["end"]) * smoothstep(p)
            if p >= 1:
                boomerang, boomerang_end = None, now

        player.target = int(round(play_idx))

        label = f"yaw {yaw_smooth if yaw_smooth is not None else 0:+.0f} | {player.names[player.target]}"
        if boomerang:
            label += "  [ROCK]"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"lean {rocks.lean:+.0f} deg  speed {rocks.speed:+.0f} deg/s", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("BlazePose", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

#!/usr/bin/env python3
"""
Slice ZED landmark CSVs and videos into per-annotation clips.

For every line in <name>_annotations.txt (format: "M:SS, label_id"):
  - Take all rows in <name>_landmarks.csv with timestamp_s in [t-1, t+2)
    -> write to data/<label_name>/<name>_<index>_<start>.csv
  - Cut <name>_video.mp4 from t-1 to t+2 (clamped to 0 if t < 1)
    -> write to zed_clips/<label_name>/<name>_<index>_<start>.mp4

Folder layout expected:
  zed_annotations/<name>_annotations.txt
  zed_data/<name>_landmarks.csv
  zed_video/<name>_video.mp4

Outputs:
  data/<label_name>/...csv
  zed_clips/<label_name>/...mp4
"""

import csv
import re
import subprocess
from pathlib import Path

# ---- Config ----
ANNOTATIONS_DIR = Path("zed_annotations")
DATA_DIR = Path("zed_data")
VIDEO_DIR = Path("zed_video")

OUT_DATA_DIR = Path("data")
OUT_CLIPS_DIR = Path("zed_clips")

CLIP_BEFORE = 1.0  # seconds before the timestamp
CLIP_AFTER = 2.0    # seconds after the timestamp

# If ffmpeg isn't on your PATH, set the full path here, e.g.:
# FFMPEG_BIN = r"C:\ffmpeg\bin\ffmpeg.exe"
FFMPEG_BIN = "ffmpeg"

LABEL_MAP = {
    "1": "stomp",
    "2": "toe_knock",
    "3": "hop",
}

TIMESTAMP_RE = re.compile(r"^\s*(\d+):(\d+(?:\.\d+)?)\s*,\s*(\S+)\s*$")


def parse_timestamp(minutes: str, seconds: str) -> float:
    return int(minutes) * 60 + float(seconds)


def parse_annotation_file(path: Path):
    """Yield (start_seconds, label_name) for each valid line."""
    entries = []
    with open(path, "r") as f:
        for lineno, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line:
                continue
            m = TIMESTAMP_RE.match(line)
            if not m:
                print(f"  [WARN] {path.name}:{lineno} - could not parse line: {raw_line!r}")
                continue
            minutes, seconds, label_raw = m.groups()
            start = parse_timestamp(minutes, seconds)
            label_name = LABEL_MAP.get(label_raw, label_raw)
            if label_raw not in LABEL_MAP:
                print(f"  [WARN] {path.name}:{lineno} - unknown label '{label_raw}', using as-is")
            entries.append((start, label_name))
    return entries


def slice_csv(landmarks_csv: Path, start: float, end: float, out_path: Path):
    with open(landmarks_csv, "r", newline="") as f_in:
        reader = csv.reader(f_in)
        header = next(reader)
        try:
            ts_idx = header.index("timestamp_s")
        except ValueError:
            raise RuntimeError(f"'timestamp_s' column not found in {landmarks_csv}")

        rows = [row for row in reader if row and start <= float(row[ts_idx]) < end]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)
        writer.writerows(rows)

    return len(rows)


def slice_video(video_path: Path, start: float, duration: float, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Re-encode for frame-accurate cuts (mp4v/H.264 keyframe issues with -c copy)
    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", f"{start:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        str(out_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print(
            f"    [ERROR] Could not find ffmpeg (tried '{FFMPEG_BIN}'). "
            "Install ffmpeg and ensure it's on PATH, or set FFMPEG_BIN "
            "at the top of this script to the full path of ffmpeg.exe."
        )
        return False
    if result.returncode != 0:
        print(f"    [ERROR] ffmpeg failed for {out_path.name}:\n{result.stderr[-1500:]}")
        return False
    return True


def fmt_time(t: float) -> str:
    return f"{t:.2f}".replace(".", "p")


def check_ffmpeg():
    try:
        subprocess.run([FFMPEG_BIN, "-version"], capture_output=True, text=True)
        return True
    except FileNotFoundError:
        return False


def main():
    if not ANNOTATIONS_DIR.exists():
        print(f"[ERROR] Annotations dir not found: {ANNOTATIONS_DIR.resolve()}")
        return

    ffmpeg_ok = check_ffmpeg()
    if not ffmpeg_ok:
        print(
            f"[WARN] ffmpeg not found (tried '{FFMPEG_BIN}'). Video clips will be skipped.\n"
            "       Install ffmpeg and put it on PATH (e.g. 'winget install ffmpeg', "
            "then open a new terminal), or set FFMPEG_BIN at the top of this script "
            "to the full path of ffmpeg.exe. CSV slicing will still run.\n"
        )

    annotation_files = sorted(ANNOTATIONS_DIR.glob("*_annotations.txt"))
    if not annotation_files:
        print(f"[ERROR] No *_annotations.txt files found in {ANNOTATIONS_DIR}")
        return

    total_csv_clips = 0
    total_video_clips = 0

    for ann_path in annotation_files:
        name = ann_path.name[: -len("_annotations.txt")]
        landmarks_csv = DATA_DIR / f"{name}_landmarks.csv"
        video_path = VIDEO_DIR / f"{name}_video.mp4"

        print(f"\n=== {name} ===")

        has_csv = landmarks_csv.exists()
        has_video = video_path.exists() and ffmpeg_ok
        if not landmarks_csv.exists():
            print(f"  [WARN] Missing landmarks csv: {landmarks_csv}")
        if not video_path.exists():
            print(f"  [WARN] Missing video: {video_path}")
        if not has_csv and not has_video:
            continue

        entries = parse_annotation_file(ann_path)
        print(f"  Parsed {len(entries)} annotation(s)")

        for idx, (start, label_name) in enumerate(entries, 1):
            clip_start = max(0.0, start - CLIP_BEFORE)
            clip_end = start + CLIP_AFTER
            clip_duration = clip_end - clip_start
            base_name = f"{name}_{idx:03d}_{fmt_time(start)}"

            if has_csv:
                out_csv = OUT_DATA_DIR / label_name / f"{base_name}.csv"
                try:
                    n_rows = slice_csv(landmarks_csv, clip_start, clip_end, out_csv)
                    print(f"  [csv]   {out_csv} ({n_rows} rows)")
                    total_csv_clips += 1
                except Exception as e:
                    print(f"  [ERROR] csv slice failed for {base_name}: {e}")

            if has_video:
                out_mp4 = OUT_CLIPS_DIR / label_name / f"{base_name}.mp4"
                ok = slice_video(video_path, clip_start, clip_duration, out_mp4)
                if ok:
                    print(f"  [video] {out_mp4}")
                    total_video_clips += 1

    print(f"\nDone. {total_csv_clips} csv clips, {total_video_clips} video clips written.")
    print(f"CSV clips in: {OUT_DATA_DIR.resolve()}")
    print(f"Video clips in: {OUT_CLIPS_DIR.resolve()}")


if __name__ == "__main__":
    main()

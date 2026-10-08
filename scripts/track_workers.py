"""
Simple anonymous person tracker with user-defined zones and a work report.

What it does
  1. Detects people (YOLO) and tracks them (ByteTrack / BoT-SORT) -> "Person #N"
  2. You draw two zones on the first frame: ENTRY/EXIT and WORKING AREA
  3. Tracks time each person spends in each zone
  4. At the end writes report.csv + prints who was WORKING / NOT WORKING

Install
  pip install ultralytics opencv-python numpy

Run
  python track_workers.py                              # pick a video in a dialog
  python track_workers.py --source office.mp4          # give the path directly
  python track_workers.py --source 0                   # webcam
  python track_workers.py --source office.mp4 --reuse-zones   # skip redrawing

Zone drawing (on the first frame of the selected video): left-click = add point, ENTER = finish polygon,
R = reset points, Q = quit.  Press Q during tracking to stop early.
No face recognition is used - only temporary anonymous IDs.
"""

import argparse
import csv
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO  # noqa: F401

ZONE_NAMES = ["ENTRY_EXIT", "WORK_AREA"]
ZONE_COLORS = {"ENTRY_EXIT": (0, 200, 255), "WORK_AREA": (0, 200, 0)}


MODEL_NAME = None  # set from --model in main()

def load_model():
    """Load YOLO, working around the PyTorch >= 2.6 "weights_only" UnpicklingError
    that older ultralytics versions hit (best fix: pip install -U ultralytics).
    Fallback: PyTorch's own TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD switch is turned on for
    this single load only. Acceptable here because the file is the official
    yolov8n.pt that ultralytics downloads itself - don't do this for unknown files."""
    from ultralytics import YOLO
    try:
        return YOLO(MODEL_NAME)
    except Exception as e:                                    # noqa: BLE001
        if "weights_only" not in str(e) and "UnpicklingError" not in repr(e):
            raise
        key = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
        previous = os.environ.get(key)
        os.environ[key] = "1"
        try:
            return YOLO(MODEL_NAME)
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def get_device(preference="auto"):
    """Pick the compute device. Returns (device_string, human_label).
    auto -> NVIDIA GPU (cuda:0) if available, else Apple GPU (mps), else CPU."""
    try:
        import torch
    except Exception:                                         # noqa: BLE001
        return "cpu", "CPU"
    if preference not in ("auto", "cpu") and preference:
        return preference, preference
    if preference != "cpu":
        if torch.cuda.is_available():
            return "cuda:0", f"GPU ({torch.cuda.get_device_name(0)})"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps", "Apple GPU (mps)"
    return "cpu", "CPU"


def pick_video():
    """Open a file dialog so the user can select a video file."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"),
                   ("All files", "*.*")],
    )
    root.destroy()
    return path


# --------------------------------------------------------------------------
# Per-person bookkeeping
# --------------------------------------------------------------------------
class PersonStats:
    def __init__(self, ts):
        self.first_seen = ts
        self.last_seen = ts
        self.zone_time = defaultdict(float)   # seconds per zone
        self.visited = []                     # zone history, duplicates collapsed
        self.trail = deque(maxlen=40)

    def update(self, ts, zone, foot, max_gap):
        dt = ts - self.last_seen
        # Only count time if the track was continuous (ignore long gaps)
        if 0 < dt <= max_gap:
            self.zone_time[zone] += dt
        self.last_seen = ts
        if not self.visited or self.visited[-1] != zone:
            self.visited.append(zone)
        self.trail.append(foot)

    @property
    def total(self):
        return self.last_seen - self.first_seen


def fmt(sec):
    sec = int(round(sec))
    return f"{sec // 60}m {sec % 60:02d}s"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None,
                    help="video path or webcam index (omit to pick a file in a dialog)")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--tracker", default="bytetrack.yaml",
                    help="bytetrack.yaml or botsort.yaml")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--device", default="auto",
                    help="auto (GPU if available), cpu, cuda:0, mps ...")
    ap.add_argument("--reuse-zones", action="store_true",
                    help="reuse the saved zones for this video instead of redrawing")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--min-seconds", type=float, default=2.0,
                    help="ignore tracks shorter than this (false detections)")
    ap.add_argument("--work-threshold", type=float, default=0.5,
                    help="fraction of time in WORK_AREA to count as WORKING")
    ap.add_argument("--max-gap", type=float, default=1.0,
                    help="seconds; larger gaps in a track are not counted as time")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    if args.source is None:
        args.source = pick_video()
        if not args.source:
            raise SystemExit("No video selected.")
    print(f"Source: {args.source}")

    is_file = not args.source.isdigit()
    cap = cv2.VideoCapture(args.source if is_file else int(args.source))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {args.source}")

    ok, first = cap.read()
    if not ok:
        raise SystemExit("Could not read first frame.")
    h, w = first.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    stem = Path(args.source).stem if is_file else "webcam"
    zones = load_or_draw_zones(first, out_dir / f"{stem}_zones.json",
                               reuse=args.reuse_zones)
    writer = cv2.VideoWriter(str(out_dir / "processed.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    global MODEL_NAME
    MODEL_NAME = args.model
    model = load_model()
    device, device_label = get_device(args.device)
    use_half = device.startswith("cuda")
    print(f"Running on: {device_label}")
    people = {}
    frame_idx = 0
    t0 = time.time()
    frame = first

    while True:
        # Video time for files (deterministic), wall-clock for webcam
        ts = frame_idx / fps if is_file else time.time() - t0

        res = model.track(frame, persist=True, classes=[0], conf=args.conf,
                          tracker=args.tracker, device=device, half=use_half,
                          verbose=False)[0]

        # Draw zones
        overlay = frame.copy()
        for name, poly in zones.items():
            cv2.fillPoly(overlay, [poly], ZONE_COLORS[name])
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
        for name, poly in zones.items():
            cv2.polylines(frame, [poly], True, ZONE_COLORS[name], 2)
            cv2.putText(frame, name, tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, ZONE_COLORS[name], 2)

        if res.boxes is not None and res.boxes.id is not None:
            ids = res.boxes.id.int().cpu().tolist()
            boxes = res.boxes.xyxy.cpu().numpy()
            for pid, (x1, y1, x2, y2) in zip(ids, boxes):
                foot = (int((x1 + x2) / 2), int(y2))     # bottom-centre = feet
                zone = zone_of(foot, zones)

                if pid not in people:
                    people[pid] = PersonStats(ts)
                people[pid].update(ts, zone, foot, args.max_gap)

                color = ZONE_COLORS.get(zone, (200, 200, 200))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"Person #{pid} [{zone}]",
                            (int(x1), max(int(y1) - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                pts = np.array(people[pid].trail, np.int32)
                if len(pts) > 1:
                    cv2.polylines(frame, [pts], False, color, 2)

        cv2.putText(frame, f"t={fmt(ts)}  tracked={len(people)}", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(frame)

        if not args.no_display:
            cv2.imshow("Tracking (Q to stop)", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break

        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # ----------------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------------
    rows = []
    for pid, p in sorted(people.items()):
        if p.total < args.min_seconds:
            continue
        work = p.zone_time["WORK_AREA"]
        entry = p.zone_time["ENTRY_EXIT"]
        other = p.zone_time["OTHER"]
        counted = work + entry + other
        ratio = work / counted if counted > 0 else 0.0
        verdict = "WORKING" if ratio >= args.work_threshold else "NOT WORKING"
        rows.append({
            "person": f"Person #{pid}",
            "first_seen": fmt(p.first_seen),
            "last_seen": fmt(p.last_seen),
            "total_time": fmt(p.total),
            "work_time": fmt(work),
            "entry_exit_time": fmt(entry),
            "other_time": fmt(other),
            "work_percent": f"{ratio * 100:.0f}%",
            "zone_history": " > ".join(p.visited),
            "verdict": verdict,
        })

    report_path = out_dir / "report.csv"
    if rows:
        with open(report_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=rows[0].keys())
            wr.writeheader()
            wr.writerows(rows)

    print("\n" + "=" * 78)
    print(f"{'PERSON':<12}{'TOTAL':<10}{'WORK':<10}{'WORK %':<9}{'VERDICT':<13}ZONES")
    print("-" * 78)
    for r in rows:
        print(f"{r['person']:<12}{r['total_time']:<10}{r['work_time']:<10}"
              f"{r['work_percent']:<9}{r['verdict']:<13}{r['zone_history']}")
    print("=" * 78)
    print(f"Report: {report_path}\nVideo : {out_dir / 'processed.mp4'}")


if __name__ == "__main__":
    main()
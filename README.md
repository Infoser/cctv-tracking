# Worker Tracker

A small web app that detects people in a video, tracks each one with a temporary
anonymous ID (Person #1, #2, ...), lets you mark an **Entry/Exit area** and a
**Work area** on the video, and produces a report showing who was working and who
was not.

No face recognition or identity lookup is used. Person photos are plain crops from the
video and are stored only on the machine running the app.

## Features
- Upload a video and draw the two areas in the browser
- Multi-person detection and tracking with stable IDs
- Per-person report: photo, ID, time in each area, work %, zone timeline, how they
  appeared and left, verdict (WORKING / NOT WORKING / BRIEF)
- Processed video with boxes, IDs, zone boundaries and movement trails
- Downloads: HTML/PDF report, Excel, CSV (summary + zone timeline), JSON, photos ZIP, video

## Tech stack
| Part | Choice |
|---|---|
| Detection | YOLOv8n (Ultralytics), person class only |
| Tracking | ByteTrack (default) or BoT-SORT |
| Zone logic | Feet position (bottom-centre of box) inside user-drawn polygons, 0.5 s debounce |
| Storage | Per-job folder with CSV / JSON / XLSX / HTML files |
| UI | Flask backend + single-page HTML/JS frontend |

## Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000. The YOLO weights download automatically on first run.

GPU: install a CUDA build of PyTorch (see pytorch.org) and the app uses it automatically.
Check with: `python -c "import torch; print(torch.cuda.is_available())"`

## How to use
1. Upload a video.
2. Click the corners of the Entry/Exit area, then of the Work area, on the first frame.
3. Start tracking, wait for processing, then read the report and download what you need.

A person counts as WORKING when at least 50 % of their tracked time is spent inside the
work area (adjustable in Advanced settings).

## Command-line version
`scripts/track_workers.py` is a simpler single-file version that opens a file picker and
OpenCV windows instead of a browser.

## Limitations
- A person who leaves the view and returns may get a new ID.
- "Working" means being inside the work area, not a measurement of activity.
- Accuracy depends on camera angle, lighting and how well the zones are drawn.
- One video is processed at a time; there is no login, so run it locally only.

## Troubleshooting
- `UnpicklingError: Weights only load failed` - run `pip install -U ultralytics`.

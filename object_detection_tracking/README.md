# Object Detection and Tracking

A Python computer-vision pipeline for real-time multi-object detection and
tracking. The project combines YOLOv8, OpenCV, the SORT tracker, and SQLite to
detect objects, assign stable track IDs, save annotated video, and produce
session summaries.

## Features

- YOLOv8 object detection through Ultralytics.
- SORT multi-object tracking with Kalman filtering and IoU association.
- Stable track IDs, class labels, confidence scores, and bounding boxes.
- Webcam and video-file input support.
- Annotated MP4 output for every run.
- Per-frame detection logging in SQLite.
- Text reports for individual sessions or all stored sessions.
- Central configuration for model, thresholds, paths, and tracker behavior.

## Pipeline

```text
Video source -> YOLOv8 detector -> SORT tracker -> annotations
                                                                                     |       |
                                                                                     v       v
                                                                            SQLite DB  output/*.mp4
```

## Project Layout

```text
task 4/
├── object_detection_tracking/
│   ├── main.py              # Detection and tracking entry point
│   ├── detector.py          # YOLOv8 detector wrapper
│   ├── sort_tracker.py      # SORT implementation
│   ├── database.py          # SQLite session and detection storage
│   ├── report.py            # Session report CLI
│   ├── config.py            # Runtime configuration
│   ├── requirements.txt     # Python dependencies
│   ├── yolov8n.pt           # YOLOv8 nano weights
│   ├── data/                # Input videos
│   ├── output/              # Annotated videos
│   └── logs/                # Runtime logs
├── run.ps1                  # PowerShell launcher for detection
└── report.ps1               # PowerShell launcher for reports
```

## Requirements

- Python 3.9 or newer.
- A webcam for live detection, or a readable video file.
- Windows users can run the included PowerShell scripts from the repository
    root. Linux and macOS users can run the Python modules directly.

## Installation

From the repository root:

```powershell
cd object_detection_tracking
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The included `yolov8n.pt` weights are used by default. If the configured
weights are missing, Ultralytics can download them on first use.

## Usage

### Webcam

Run from inside `object_detection_tracking`:

```bash
python main.py
```

Or run from the repository root in PowerShell:

```powershell
.\run.ps1
```

Press `q` in the OpenCV window to stop the run.

### Video file

```bash
python main.py --source data/sample.mp4
```

The `--source` value can be a webcam index such as `0` or a path to a video
file. The processed video is written to `output/`.

### Command-line options

```text
--source SOURCE    Webcam index or video-file path
--weights WEIGHTS  YOLO weights, for example yolov8s.pt
--no-db            Disable SQLite logging for the current run
```

Examples:

```bash
python main.py --source data/sample.mp4 --weights yolov8s.pt
python main.py --source 0 --no-db
```

## Reports

Each database-enabled run creates a session in `detections.db`. To print the
most recent session report:

```bash
python report.py
```

Other report commands:

```bash
python report.py --session 3
python report.py --all-sessions
python report.py --db path/to/custom.db
```

From the repository root, the equivalent PowerShell commands are:

```powershell
.\report.ps1
.\report.ps1 --all-sessions
```

Reports include session metadata, per-track summaries, and per-class
aggregates.

## Configuration

Edit `object_detection_tracking/config.py` to change defaults:

| Setting | Default | Purpose |
| --- | --- | --- |
| `VIDEO_SOURCE` | `0` | Webcam index or default video path |
| `YOLO_WEIGHTS` | `yolov8n.pt` | Detection model weights |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum detection confidence |
| `MAX_AGE` | `30` | Frames before an unmatched track is removed |
| `MIN_HITS` | `3` | Matches needed to confirm a track |
| `IOU_MATCH_THRESHOLD` | `0.3` | Minimum IoU for detection-track matching |
| `LOG_TO_DB` | `True` | Enable or disable SQLite logging |
| `DB_PATH` | `detections.db` | SQLite database path |
| `OUTPUT_DIR` | `output` | Annotated-video directory |
| `LOGS_DIR` | `logs` | Runtime-log directory |

## Generated Files

- `output/output_*.mp4`: annotated video from a run.
- `logs/run_*.log`: application logs.
- `detections.db`: sessions, detections, and track summaries.

These files are useful for local results and demonstrations. For a clean
production deployment, keep large generated media outside source control.

## Troubleshooting

- If the webcam cannot be opened, try another `--source` index or use a video
    file.
- If a model cannot be found, provide a valid path with `--weights` or allow
    Ultralytics to download the selected model.
- If no report data exists, run `python main.py` without `--no-db` first.
- On Windows PowerShell, activate the virtual environment before running the
    launcher scripts.

## License

MIT

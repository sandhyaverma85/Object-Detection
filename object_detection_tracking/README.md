# Object Detection & Tracking

Real-time multi-object detection (YOLOv8) + SORT tracking + SQLite logging,
written in Python with OpenCV.

---

## Project Structure

```
object_detection_tracking/
├── main.py          # Pipeline entry point
├── detector.py      # YOLOv8 wrapper (Ultralytics)
├── sort_tracker.py  # SORT algorithm (Kalman + Hungarian)
├── database.py      # SQLite DetectionDB class
├── report.py        # Post-session summary CLI
├── config.py        # All tunable settings
├── requirements.txt
├── models/          # YOLO weight files (.pt)
├── data/            # Input video files
├── output/          # Annotated output videos
└── logs/            # Run logs
```

---

## Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `ultralytics` downloads `yolov8n.pt` automatically on first run if
> the file is not already present in the working directory or `models/`.

### 2 — Run on webcam (default)

```bash
python main.py
```

Press **`q`** in the video window to stop.

### 3 — Run on a video file

```bash
python main.py --source data/sample.mp4
```

### 4 — View the session report

```bash
python report.py
```

Additional report options:

```bash
python report.py --session 3        # specific session ID
python report.py --all-sessions     # every session in the DB
python report.py --db /path/to/custom.db
```

---

## Configuration (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `VIDEO_SOURCE` | `0` | Webcam index or video file path |
| `YOLO_WEIGHTS` | `"yolov8n.pt"` | Model size: n / s / m / l / x |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum detection score |
| `MAX_AGE` | `30` | Frames before a stale track is deleted |
| `MIN_HITS` | `3` | Matches before a track is confirmed |
| `IOU_MATCH_THRESHOLD` | `0.3` | IoU required for detection↔track association |
| `LOG_TO_DB` | `True` | Toggle SQLite logging |
| `DB_PATH` | `"detections.db"` | Database file path |
| `OUTPUT_DIR` | `"output"` | Directory for annotated video files |

---

## Switching from SORT to Deep SORT

SORT uses only bounding-box position for re-identification.  Deep SORT adds a
learned appearance embedding (Re-ID CNN) so the tracker can re-associate an
object after occlusion or long absence.

### Install

```bash
pip install deep-sort-realtime
```

### Minimal swap in `main.py`

Replace the `Sort` import and initialisation:

```python
# --- BEFORE (SORT) ---
from sort_tracker import Sort
tracker = Sort(
    max_age=config.MAX_AGE,
    min_hits=config.MIN_HITS,
    iou_threshold=config.IOU_MATCH_THRESHOLD,
)

# --- AFTER (Deep SORT) ---
from deep_sort_realtime.deepsort_tracker import DeepSort
tracker = DeepSort(
    max_age=config.MAX_AGE,
    n_init=config.MIN_HITS,
    nn_budget=100,           # max appearance descriptors per track
    embedder="mobilenet",    # CNN backbone: mobilenet | torchreid | clip_ViT-B/16 | …
    half=True,               # FP16 inference (faster on CUDA)
)
```

Replace the per-frame tracking call:

```python
# --- BEFORE (SORT) ---
# det_array shape: (N, 5)  [x1, y1, x2, y2, conf]
tracked = tracker.update(det_array)
# tracked shape:  (M, 6)  [x1, y1, x2, y2, conf, track_id]

# --- AFTER (Deep SORT) ---
# deep-sort-realtime expects a list of ([x1,y1,w,h], conf, class_name) tuples
# plus the raw frame for the appearance embedder.
ds_detections = [
    ([d["bbox"][0], d["bbox"][1],
      d["bbox"][2] - d["bbox"][0],   # width
      d["bbox"][3] - d["bbox"][1]],  # height
     d["confidence"],
     d["class_name"])
    for d in raw_detections
]
ds_tracks = tracker.update_tracks(ds_detections, frame=frame)

# Convert back to the same (M, 6) format the rest of the pipeline expects
tracked_rows = []
for t in ds_tracks:
    if not t.is_confirmed():
        continue
    l, top, r, b = t.to_ltrb()
    tracked_rows.append([l, top, r, b, 1.0, t.track_id])
tracked = np.array(tracked_rows, dtype=float) if tracked_rows else np.empty((0, 6))
```

No other files need to change — `database.py`, `report.py`, and `config.py`
are all tracker-agnostic.

---

## Database Schema

```sql
-- Active sessions
CREATE TABLE sessions (
    session_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    video_source TEXT,
    started_at   TEXT,
    ended_at     TEXT
);

-- Per-frame detections
CREATE TABLE detections (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   INTEGER REFERENCES sessions(session_id),
    frame_number INTEGER,
    timestamp    TEXT,
    track_id     INTEGER,
    class_id     INTEGER,
    class_name   TEXT,
    confidence   REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL
);

-- Continuously upserted track statistics
CREATE TABLE track_summary (
    session_id        INTEGER,
    track_id          INTEGER,
    class_name        TEXT,
    first_seen_frame  INTEGER,
    last_seen_frame   INTEGER,
    total_frames_seen INTEGER,
    avg_confidence    REAL,
    PRIMARY KEY (session_id, track_id)
);
```

---

## License

MIT

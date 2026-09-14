# =============================================================================
# config.py — Centralised configuration for object_detection_tracking
# All tunable parameters live here so no other file needs to be edited for
# common adjustments (source, model size, thresholds, …).
# =============================================================================

# ---------------------------------------------------------------------------
# Video source
#   0         → default webcam
#   str path  → video file (e.g. "data/sample.mp4")
# ---------------------------------------------------------------------------
VIDEO_SOURCE = 0

# ---------------------------------------------------------------------------
# YOLO model weights
#   "yolov8n.pt"  nano   – fastest,  least accurate
#   "yolov8s.pt"  small
#   "yolov8m.pt"  medium
#   "yolov8l.pt"  large
#   "yolov8x.pt"  xlarge – slowest, most accurate
# The Ultralytics library downloads the file automatically on first use.
# ---------------------------------------------------------------------------
YOLO_WEIGHTS = "yolov8n.pt"

# ---------------------------------------------------------------------------
# Detection confidence threshold (0.0 – 1.0).
# Detections below this value are discarded before tracking.
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# SORT tracker parameters
#   MAX_AGE          – number of consecutive unmatched frames before a track
#                      is deleted
#   MIN_HITS         – minimum number of matched frames before a track is
#                      considered confirmed and rendered
#   IOU_MATCH_THRESHOLD – minimum IoU to associate a detection to a track
# ---------------------------------------------------------------------------
MAX_AGE = 30
MIN_HITS = 3
IOU_MATCH_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# Database / logging toggle
#   True  → write every detection to detections.db
#   False → run without any DB I/O (useful for pure-speed benchmarks)
# ---------------------------------------------------------------------------
LOG_TO_DB = True

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
DB_PATH = "detections.db"
OUTPUT_DIR = "output"
LOGS_DIR = "logs"
MODELS_DIR = "models"

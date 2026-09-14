# =============================================================================
# main.py — Entry point for real-time object detection + multi-object tracking
#
# Pipeline per frame:
#   VideoCapture → Detector (YOLOv8) → Sort tracker → DB logger
#                                                   → annotated display + video write
#
# Usage:
#   python main.py                          # webcam (index 0)
#   python main.py --source data/sample.mp4 # video file
#   python main.py --source 0 --no-db       # skip DB logging
# =============================================================================

from __future__ import annotations

import argparse
import os
import sys
import time
import datetime
import logging

import cv2
import numpy as np

import config
from detector import Detector
from sort_tracker import Sort, _iou as _box_iou
from database import DetectionDB


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

os.makedirs(config.LOGS_DIR, exist_ok=True)
log_filename = os.path.join(
    config.LOGS_DIR,
    f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette — one stable BGR colour per track ID
# ---------------------------------------------------------------------------

_PALETTE = [
    (255,  56,  56), (255, 157,  51), (255, 112,  31), (255, 178,  29),
    ( 30, 207,  98), (  0, 205, 205), ( 38,  36, 212), (148,   0, 211),
    (255,   0, 255), (  0, 128, 255), (102, 205, 170), (255, 165,   0),
]


def _track_colour(track_id: int):
    """Return a stable BGR colour for the given track ID."""
    return _PALETTE[int(track_id) % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Annotation helper
# ---------------------------------------------------------------------------

def _annotate_frame(
    frame: np.ndarray,
    tracked: np.ndarray,
    detections_meta: list,
    tracker: Sort,
) -> np.ndarray:
    """
    Draw bounding boxes, class labels, confidence scores, and track IDs.

    Parameters
    ----------
    frame           : raw BGR frame
    tracked         : (M, 6) array  [x1,y1,x2,y2, conf, track_id]
    detections_meta : original detection dicts (class_id, class_name, conf)
                      used to recover class info matched to each track
    tracker         : Sort instance (for the ID→class mapping built below)
    """
    annotated = frame.copy()

    for row in tracked:
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        conf     = row[4]
        track_id = int(row[5])

        colour = _track_colour(track_id)

        # --- bounding box ---
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

        # --- label --------------------------------------------------------
        # class name comes from tracker._id_to_meta dict populated below
        class_name = tracker._id_to_meta.get(track_id, {}).get("class_name", "?")
        label = f"#{track_id} {class_name} {conf:.2f}"

        # Background rectangle for readability
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        top_left     = (x1, max(y1 - th - baseline - 4, 0))
        bottom_right = (x1 + tw + 4, max(y1, th + baseline + 4))
        cv2.rectangle(annotated, top_left, bottom_right, colour, cv2.FILLED)
        cv2.putText(
            annotated, label,
            (x1 + 2, max(y1 - baseline - 2, th + 2)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # --- frame counter (top-left) ---
    cv2.putText(
        annotated,
        f"Tracks: {len(tracked)}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
    )

    return annotated


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time object detection and multi-object tracking"
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Video source: integer webcam index or path to a video file. "
             "Overrides VIDEO_SOURCE in config.py.",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Disable database logging for this run.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="YOLO weights to use (e.g. yolov8s.pt). Overrides YOLO_WEIGHTS in config.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Resolve config overrides ----------------------------------------
    video_source = args.source if args.source is not None else config.VIDEO_SOURCE
    # If the source looks like an integer, convert it
    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)

    log_to_db  = config.LOG_TO_DB and not args.no_db
    weights    = args.weights if args.weights else config.YOLO_WEIGHTS

    logger.info("=== Object Detection & Tracking Session Start ===")
    logger.info("Source   : %s", video_source)
    logger.info("Weights  : %s", weights)
    logger.info("DB log   : %s", log_to_db)

    # --- Initialise components -------------------------------------------
    detector = Detector(
        weights_path=weights,
        confidence_threshold=config.CONFIDENCE_THRESHOLD,
    )
    tracker = Sort(
        max_age=config.MAX_AGE,
        min_hits=config.MIN_HITS,
        iou_threshold=config.IOU_MATCH_THRESHOLD,
    )
    # Attach a mutable metadata dict to the tracker for annotating by track ID
    tracker._id_to_meta: dict = {}

    # --- Video capture ---------------------------------------------------
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error("Cannot open video source: %s", video_source)
        sys.exit(1)

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src      = cap.get(cv2.CAP_PROP_FPS) or 30.0

    logger.info("Frame size: %dx%d @ %.1f fps", frame_width, frame_height, fps_src)

    # --- Output video writer ---------------------------------------------
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_filename = os.path.join(
        config.OUTPUT_DIR,
        f"output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_filename, fourcc, fps_src, (frame_width, frame_height))
    logger.info("Writing output to: %s", out_filename)

    # --- Database --------------------------------------------------------
    db: DetectionDB | None = None
    if log_to_db:
        db = DetectionDB(config.DB_PATH)
        db.connect()
        db.start_session(str(video_source))
        logger.info("DB session %d opened in: %s", db.session_id, config.DB_PATH)

    # --- Main loop -------------------------------------------------------
    frame_number = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream.")
                break

            frame_number += 1

            # 1. Detect
            raw_detections = detector.detect(frame)

            # 2. Build (N,5) array for SORT
            det_array = detector.detections_to_array(raw_detections)

            # 3. Track
            tracked = tracker.update(det_array)   # (M, 6): x1,y1,x2,y2,conf,id

            # 4. Update class metadata for annotator / DB
            #    Match tracked boxes back to detections by best IoU
            tracked_objects: list = []
            for row in tracked:
                track_id = int(row[5])
                trk_box  = row[:4]

                # Find best-matching original detection to get class info
                best_class_id   = -1
                best_class_name = "unknown"
                best_conf       = float(row[4])
                best_iou        = -1.0

                for d in raw_detections:
                    iou_val = _box_iou(trk_box, np.array(d["bbox"]))
                    if iou_val > best_iou:
                        best_iou        = iou_val
                        best_class_id   = d["class_id"]
                        best_class_name = d["class_name"]
                        best_conf       = d["confidence"]

                tracker._id_to_meta[track_id] = {
                    "class_id":   best_class_id,
                    "class_name": best_class_name,
                }

                tracked_objects.append(
                    {
                        "track_id":   track_id,
                        "class_id":   best_class_id,
                        "class_name": best_class_name,
                        "confidence": best_conf,
                        "x1": float(row[0]),
                        "y1": float(row[1]),
                        "x2": float(row[2]),
                        "y2": float(row[3]),
                    }
                )

            # 5. Log to DB
            if db and tracked_objects:
                db.log_detections(frame_number, tracked_objects)

            # 6. Annotate + display + write
            annotated = _annotate_frame(frame, tracked, raw_detections, tracker)
            writer.write(annotated)
            cv2.imshow("Object Detection & Tracking", annotated)

            # 'q' → quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("User requested quit.")
                break

    finally:
        elapsed = time.time() - t_start
        avg_fps = frame_number / elapsed if elapsed > 0 else 0
        logger.info(
            "Processed %d frames in %.1fs (%.1f fps avg)",
            frame_number, elapsed, avg_fps,
        )

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        if db:
            db.close()
            logger.info("DB session closed.")

        logger.info("Output saved: %s", out_filename)
        logger.info("=== Session End ===")


if __name__ == "__main__":
    main()

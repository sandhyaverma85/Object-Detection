# =============================================================================
# detector.py — YOLOv8 inference wrapper
#
# Wraps Ultralytics' YOLO class to expose a clean, frame-level interface.
# The rest of the pipeline only ever calls Detector.detect(frame) and
# receives a plain list of dicts — no Ultralytics types leak out.
# =============================================================================

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any
from ultralytics import YOLO


class Detector:
    """
    Thin wrapper around a YOLOv8 model.

    Parameters
    ----------
    weights_path        : path/name of the YOLO weights file
                          (e.g. "yolov8n.pt" — downloaded automatically on
                          first use if not present locally)
    confidence_threshold: detections below this score are discarded
    device              : inference device — "" lets Ultralytics choose
                          automatically (CUDA if available, else CPU)
    """

    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: str = "",
    ) -> None:
        self.confidence_threshold = confidence_threshold

        # Load the model — Ultralytics prints a one-time download message
        # when the weights are not cached locally.
        self.model = YOLO(weights_path)
        self._device = device

        # Build an id→name mapping for fast class-name lookup
        self._class_names: Dict[int, str] = self.model.names  # dict[int, str]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on a single BGR frame (as returned by cv2.VideoCapture).

        Returns
        -------
        list of dicts, one per detection that passes the confidence threshold:
            {
                "bbox"      : [x1, y1, x2, y2]   (float, pixel coords),
                "confidence": float,
                "class_id"  : int,
                "class_name": str,
            }
        """
        # Run YOLO inference (returns a list with one Results object per image)
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            device=self._device,
            verbose=False,          # suppress per-frame console output
        )

        detections: List[Dict[str, Any]] = []

        # results[0] because we always pass a single frame
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue  # secondary check (model's own conf filter may be lenient)

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            class_name = self._class_names.get(class_id, str(class_id))

            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": class_id,
                    "class_name": class_name,
                }
            )

        return detections

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def detections_to_array(
        self, detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Convert the list returned by detect() into the (N, 5) NumPy array
        expected by Sort.update():  [x1, y1, x2, y2, confidence].

        Returns an empty (0, 5) array when *detections* is empty.
        """
        if not detections:
            return np.empty((0, 5), dtype=float)

        rows = [
            [*d["bbox"], d["confidence"]]
            for d in detections
        ]
        return np.array(rows, dtype=float)

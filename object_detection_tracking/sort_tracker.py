# =============================================================================
# sort_tracker.py — Pure-Python SORT (Simple Online and Realtime Tracking)
#
# Implements:
#   KalmanBoxTracker  – one per tracked object; wraps filterpy's KalmanFilter
#                       with a constant-velocity bounding-box motion model.
#   Sort              – manages the full set of tracks; on each frame it:
#                         1. Predicts every track's new position.
#                         2. Matches predictions to incoming detections using
#                            the Hungarian algorithm on an IoU cost matrix.
#                         3. Updates matched tracks, creates new ones for
#                            unmatched detections, and ages/removes stale tracks.
#
# Reference: Bewley et al., "Simple Online and Realtime Tracking" (2016)
#            https://arxiv.org/abs/1602.00763
# =============================================================================

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Bounding-box utility helpers
# ---------------------------------------------------------------------------

def _box_to_z(bbox: np.ndarray) -> np.ndarray:
    """
    Convert [x1, y1, x2, y2] to the Kalman state vector
    [cx, cy, s, r]^T  where
        cx, cy  = centre
        s       = area  (scale)
        r       = aspect ratio (width / height) — kept constant
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h          # area
    r = w / float(h) if h != 0 else 1.0
    return np.array([cx, cy, s, r], dtype=float).reshape((4, 1))


def _z_to_box(z: np.ndarray) -> np.ndarray:
    """
    Inverse of _box_to_z: state vector → [x1, y1, x2, y2].
    z may be shape (4,) or (4,1) — flatten first.
    """
    z = z.flatten()
    cx, cy, s, r = float(z[0]), float(z[1]), float(z[2]), float(z[3])
    if s <= 0 or r <= 0:
        # Degenerate state — return a zero-size box at the centre
        return np.array([cx, cy, cx, cy], dtype=float)
    w = np.sqrt(s * r)
    h = s / w
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=float)


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Intersection-over-Union between two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0.0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _iou_matrix(
    trackers: List[np.ndarray],
    detections: List[np.ndarray],
) -> np.ndarray:
    """Build an N×M IoU matrix between *trackers* and *detections*."""
    iou_mat = np.zeros((len(trackers), len(detections)), dtype=float)
    for t, trk in enumerate(trackers):
        for d, det in enumerate(detections):
            iou_mat[t, d] = _iou(trk, det)
    return iou_mat


def _hungarian_match(
    iou_mat: np.ndarray,
    iou_threshold: float,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    Apply the Hungarian algorithm to the IoU cost matrix.

    Returns
    -------
    matches        : Kx2 array of (tracker_idx, detection_idx) pairs
    unmatched_dets : list of unmatched detection indices
    unmatched_trks : list of unmatched tracker indices
    """
    if iou_mat.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            list(range(iou_mat.shape[1])),
            list(range(iou_mat.shape[0])),
        )

    # scipy minimises, so negate IoU to maximise it
    row_ind, col_ind = linear_sum_assignment(-iou_mat)
    matched_pairs = np.stack([row_ind, col_ind], axis=1)

    # Remove pairs whose IoU is below the threshold
    valid = iou_mat[row_ind, col_ind] >= iou_threshold
    matches = matched_pairs[valid]

    matched_trk_set = set(matches[:, 0].tolist()) if len(matches) else set()
    matched_det_set = set(matches[:, 1].tolist()) if len(matches) else set()

    unmatched_trks = [t for t in range(iou_mat.shape[0]) if t not in matched_trk_set]
    unmatched_dets = [d for d in range(iou_mat.shape[1]) if d not in matched_det_set]

    return matches, unmatched_dets, unmatched_trks


# ---------------------------------------------------------------------------
# Kalman bounding-box tracker
# ---------------------------------------------------------------------------

class KalmanBoxTracker:
    """
    Tracks a single object with a Kalman filter.

    State vector  x = [cx, cy, s, r, vcx, vcy, vs]^T
        cx, cy : centre coordinates
        s      : scale (area)
        r      : aspect ratio (constant — no velocity)
        vcx, vcy, vs : velocities for cx, cy, s
    """

    _count = 0  # class-level counter → persistent track IDs across all instances

    def __init__(self, bbox: np.ndarray) -> None:
        KalmanBoxTracker._count += 1
        self.id = KalmanBoxTracker._count

        # --- Kalman filter setup (7-dim state, 4-dim measurement) ----------
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # State transition matrix (constant-velocity model)
        # x_{t+1} = F * x_t
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],   # cx  += vcx
            [0, 1, 0, 0, 0, 1, 0],   # cy  += vcy
            [0, 0, 1, 0, 0, 0, 1],   # s   += vs
            [0, 0, 0, 1, 0, 0, 0],   # r    = r  (constant)
            [0, 0, 0, 0, 1, 0, 0],   # vcx += 0
            [0, 0, 0, 0, 0, 1, 0],   # vcy += 0
            [0, 0, 0, 0, 0, 0, 1],   # vs  += 0
        ], dtype=float)

        # Measurement function: observe [cx, cy, s, r] from state
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype=float)

        # Measurement noise covariance
        self.kf.R[2:, 2:] *= 10.0

        # Covariance matrix — inflate velocity components' initial uncertainty
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0

        # Process noise
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        # Initialise state from the first detection
        self.kf.x[:4] = _box_to_z(bbox)

        self.time_since_update = 0  # frames since last matched detection
        self.hit_streak = 0         # consecutive matched frames
        self.hits = 0               # total matched frames
        self.age = 0                # total frames this tracker has existed
        self.last_confidence = 0.0  # confidence score from the most recent update

    # ------------------------------------------------------------------

    def predict(self) -> np.ndarray:
        """Advance the Kalman filter one time step and return the predicted box."""
        # Prevent negative scale
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] = 0.0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return _z_to_box(self.kf.x[:4])

    def update(self, bbox: np.ndarray, confidence: float = 0.0) -> None:
        """Correct the Kalman filter state with a new measurement."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.last_confidence = confidence
        self.kf.update(_box_to_z(bbox))

    def get_state(self) -> np.ndarray:
        """Return the current best-estimate bounding box [x1,y1,x2,y2]."""
        return _z_to_box(self.kf.x[:4])


# ---------------------------------------------------------------------------
# SORT — manages all active KalmanBoxTrackers
# ---------------------------------------------------------------------------

class Sort:
    """
    SORT multi-object tracker.

    Parameters
    ----------
    max_age           : frames a track survives without any detection match
    min_hits          : matches needed before a track is considered confirmed
    iou_threshold     : minimum IoU for a valid detection↔track match
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

        # Reset the global track-ID counter so IDs start at 1 each run
        KalmanBoxTracker._count = 0

    def update(
        self,
        detections: np.ndarray,
    ) -> np.ndarray:
        """
        Run one frame of SORT.

        Parameters
        ----------
        detections : np.ndarray, shape (N, 5)
            Each row is [x1, y1, x2, y2, confidence].
            Pass an empty array (shape (0,5)) when there are no detections.

        Returns
        -------
        np.ndarray, shape (M, 6)
            Each row is [x1, y1, x2, y2, confidence, track_id] for every
            *confirmed* (hit_streak >= min_hits OR recently seen) track.
        """
        self.frame_count += 1

        # 1. Predict new positions for all existing trackers
        predicted_boxes = []
        to_del = []
        for i, trk in enumerate(self.trackers):
            box = trk.predict()
            if np.any(np.isnan(box)):
                to_del.append(i)
            else:
                predicted_boxes.append(box)

        # Remove NaN trackers in reverse order so indices remain valid
        for i in reversed(to_del):
            self.trackers.pop(i)

        # 2. Match predictions to detections via IoU + Hungarian
        if len(detections) > 0 and len(predicted_boxes) > 0:
            iou_mat = _iou_matrix(predicted_boxes, detections[:, :4])
            matches, unmatched_dets, unmatched_trks = _hungarian_match(
                iou_mat, self.iou_threshold
            )
        else:
            matches = np.empty((0, 2), dtype=int)
            unmatched_dets = list(range(len(detections)))
            unmatched_trks = list(range(len(self.trackers)))

        # 3. Update matched trackers (pass confidence so it is stored on the track)
        for t_idx, d_idx in matches:
            conf = float(detections[d_idx, 4]) if detections.shape[1] > 4 else 0.0
            self.trackers[t_idx].update(detections[d_idx, :4], confidence=conf)

        # 4. Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[d_idx, :4]))

        # 5. Collect results and remove dead tracks
        results = []
        trackers_alive = []
        for trk in self.trackers:
            if trk.time_since_update <= self.max_age:
                # Only emit confirmed tracks (or very recently matched ones)
                if trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits:
                    box = trk.get_state()
                    # Use the confidence stored on the tracker from its last update.
                    # Coasting tracks (unmatched) keep the last observed confidence.
                    results.append(np.append(box, [trk.last_confidence, trk.id]))
                trackers_alive.append(trk)

        self.trackers = trackers_alive

        return np.array(results, dtype=float) if results else np.empty((0, 6), dtype=float)

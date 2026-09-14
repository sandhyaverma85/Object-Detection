# =============================================================================
# database.py — SQLite persistence layer for object_detection_tracking
#
# Provides DetectionDB, a context-manager-friendly class that:
#   • Creates the three-table schema on first use
#   • Opens/closes sessions (rows in `sessions`)
#   • Bulk-inserts per-frame detections into `detections`
#   • Upserts running track statistics into `track_summary`
#   • Exposes summary queries used by report.py
# =============================================================================

import sqlite3
import datetime
from typing import List, Dict, Any, Optional


class DetectionDB:
    """Thin wrapper around a SQLite connection for detection logging."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self, db_path: str = "detections.db") -> None:
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.session_id: Optional[int] = None

    # Support `with DetectionDB(...) as db:` usage
    def __enter__(self) -> "DetectionDB":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
        return False  # do not suppress exceptions

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open the database connection and ensure the schema exists."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")   # better write concurrency
        self.conn.execute("PRAGMA synchronous=NORMAL;") # faster without full-sync
        self._create_schema()

    def close(self) -> None:
        """Flush any pending transaction and close the connection."""
        if self.conn:
            self.end_session()
            self.conn.close()
            self.conn = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_schema(self) -> None:
        """Create all tables if they do not yet exist."""
        ddl = """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            video_source TEXT,
            started_at   TEXT,
            ended_at     TEXT
        );

        CREATE TABLE IF NOT EXISTS detections (
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

        CREATE TABLE IF NOT EXISTS track_summary (
            session_id        INTEGER,
            track_id          INTEGER,
            class_name        TEXT,
            first_seen_frame  INTEGER,
            last_seen_frame   INTEGER,
            total_frames_seen INTEGER,
            avg_confidence    REAL,
            PRIMARY KEY (session_id, track_id)
        );
        """
        self.conn.executescript(ddl)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def start_session(self, video_source: str) -> int:
        """
        Insert a new session row and return the auto-assigned session_id.
        Must be called once before logging any detections.
        """
        now = datetime.datetime.utcnow().isoformat()
        cur = self.conn.execute(
            "INSERT INTO sessions (video_source, started_at) VALUES (?, ?)",
            (str(video_source), now),
        )
        self.conn.commit()
        self.session_id = cur.lastrowid
        return self.session_id

    def end_session(self) -> None:
        """Stamp the ended_at timestamp on the active session (if any)."""
        if self.conn and self.session_id is not None:
            now = datetime.datetime.utcnow().isoformat()
            self.conn.execute(
                "UPDATE sessions SET ended_at = ? WHERE session_id = ?",
                (now, self.session_id),
            )
            self.conn.commit()

    # ------------------------------------------------------------------
    # Per-frame logging
    # ------------------------------------------------------------------

    def log_detections(
        self,
        frame_number: int,
        tracked_objects: List[Dict[str, Any]],
    ) -> None:
        """
        Insert one row per tracked object into `detections` and upsert
        `track_summary`.

        Each item in *tracked_objects* must be a dict with keys:
            track_id, class_id, class_name, confidence, x1, y1, x2, y2
        """
        if self.conn is None or self.session_id is None:
            raise RuntimeError("Call start_session() before log_detections().")

        timestamp = datetime.datetime.utcnow().isoformat()
        sid = self.session_id

        det_rows = [
            (
                sid,
                frame_number,
                timestamp,
                obj["track_id"],
                obj["class_id"],
                obj["class_name"],
                obj["confidence"],
                obj["x1"],
                obj["y1"],
                obj["x2"],
                obj["y2"],
            )
            for obj in tracked_objects
        ]

        self.conn.executemany(
            """INSERT INTO detections
               (session_id, frame_number, timestamp, track_id, class_id,
                class_name, confidence, x1, y1, x2, y2)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            det_rows,
        )

        # Upsert track_summary — one row per (session_id, track_id)
        for obj in tracked_objects:
            self.conn.execute(
                """INSERT INTO track_summary
                   (session_id, track_id, class_name,
                    first_seen_frame, last_seen_frame,
                    total_frames_seen, avg_confidence)
                   VALUES (?, ?, ?, ?, ?, 1, ?)
                   ON CONFLICT(session_id, track_id) DO UPDATE SET
                       last_seen_frame   = excluded.last_seen_frame,
                       total_frames_seen = total_frames_seen + 1,
                       avg_confidence    = (
                           avg_confidence * total_frames_seen + excluded.avg_confidence
                       ) / (total_frames_seen + 1)
                """,
                (
                    sid,
                    obj["track_id"],
                    obj["class_name"],
                    frame_number,
                    frame_number,
                    obj["confidence"],
                ),
            )

        self.conn.commit()

    # ------------------------------------------------------------------
    # Summary queries (used by report.py)
    # ------------------------------------------------------------------

    def get_sessions(self) -> List[Dict[str, Any]]:
        """Return all session rows as a list of dicts."""
        cur = self.conn.execute(
            "SELECT session_id, video_source, started_at, ended_at FROM sessions"
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_track_summary(self, session_id: int) -> List[Dict[str, Any]]:
        """Return per-track summary rows for the given session."""
        cur = self.conn.execute(
            """SELECT track_id, class_name, first_seen_frame,
                      last_seen_frame, total_frames_seen, avg_confidence
               FROM track_summary
               WHERE session_id = ?
               ORDER BY track_id""",
            (session_id,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_class_summary(self, session_id: int) -> List[Dict[str, Any]]:
        """Return per-class aggregate statistics for the given session."""
        cur = self.conn.execute(
            """SELECT class_name,
                      COUNT(DISTINCT track_id)     AS unique_tracks,
                      COUNT(*)                     AS total_detections,
                      ROUND(AVG(avg_confidence), 4) AS mean_confidence
               FROM track_summary
               WHERE session_id = ?
               GROUP BY class_name
               ORDER BY total_detections DESC""",
            (session_id,),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

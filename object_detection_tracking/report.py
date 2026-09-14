# =============================================================================
# report.py — CLI summary report for a completed detection/tracking session
#
# Connects to the SQLite database written by main.py and prints:
#   1. A list of all sessions stored in the DB.
#   2. For the most recent session (or --session N), a per-track summary.
#   3. A per-class aggregate summary.
#
# Usage:
#   python report.py                  # report on most-recent session
#   python report.py --session 3      # report on session ID 3
#   python report.py --all-sessions   # list all sessions, then report each
# =============================================================================

from __future__ import annotations

import argparse
import sys
import os

# Allow running from any working directory
sys.path.insert(0, os.path.dirname(__file__))

import config
from database import DetectionDB


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_SEP = "-" * 72


def _print_sessions(sessions: list) -> None:
    if not sessions:
        print("  (no sessions found)")
        return
    print(f"  {'ID':>4}  {'Video Source':<30}  {'Started At':<26}  {'Ended At'}")
    print(f"  {'-'*4}  {'-'*30}  {'-'*26}  {'-'*26}")
    for s in sessions:
        print(
            f"  {s['session_id']:>4}  {str(s['video_source']):<30}  "
            f"{str(s['started_at']):<26}  {str(s['ended_at'])}"
        )


def _print_track_summary(rows: list, session_id: int) -> None:
    if not rows:
        print("  (no tracks recorded)")
        return
    print(
        f"  {'TrackID':>7}  {'Class':<20}  {'First':>6}  {'Last':>6}  "
        f"{'Frames':>7}  {'AvgConf':>8}"
    )
    print(
        f"  {'-'*7}  {'-'*20}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*8}"
    )
    for r in rows:
        print(
            f"  {r['track_id']:>7}  {r['class_name']:<20}  "
            f"{r['first_seen_frame']:>6}  {r['last_seen_frame']:>6}  "
            f"{r['total_frames_seen']:>7}  {r['avg_confidence']:>8.4f}"
        )


def _print_class_summary(rows: list, session_id: int) -> None:
    if not rows:
        print("  (no class data)")
        return
    print(
        f"  {'Class':<22}  {'UniqueT':>8}  {'TotalDet':>9}  {'MeanConf':>9}"
    )
    print(f"  {'-'*22}  {'-'*8}  {'-'*9}  {'-'*9}")
    for r in rows:
        print(
            f"  {r['class_name']:<22}  {r['unique_tracks']:>8}  "
            f"{r['total_detections']:>9}  {r['mean_confidence']:>9.4f}"
        )


def _report_session(db: DetectionDB, session: dict) -> None:
    sid = session["session_id"]
    print()
    print(_SEP)
    print(f"SESSION {sid}  |  source: {session['video_source']}")
    print(f"  Started : {session['started_at']}")
    print(f"  Ended   : {session['ended_at']}")
    print(_SEP)

    track_rows = db.get_track_summary(sid)
    print(f"\n  Per-Track Summary  ({len(track_rows)} unique tracks)")
    _print_track_summary(track_rows, sid)

    class_rows = db.get_class_summary(sid)
    print(f"\n  Per-Class Summary  ({len(class_rows)} classes)")
    _print_class_summary(class_rows, sid)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a summary report from the detection/tracking database."
    )
    parser.add_argument(
        "--session",
        type=int,
        default=None,
        metavar="N",
        help="Report on session ID N (default: most recent session).",
    )
    parser.add_argument(
        "--all-sessions",
        action="store_true",
        help="Print a report for every session in the database.",
    )
    parser.add_argument(
        "--db",
        default=config.DB_PATH,
        metavar="PATH",
        help=f"Path to the SQLite database file (default: {config.DB_PATH}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.db):
        print(f"[ERROR] Database not found: {args.db}")
        print("  Run `python main.py` first to generate tracking data.")
        sys.exit(1)

    with DetectionDB(args.db) as db:
        sessions = db.get_sessions()

        print("=" * 72)
        print("  OBJECT DETECTION & TRACKING  --  SESSION REPORT")
        print("=" * 72)
        print(f"\n  Database : {os.path.abspath(args.db)}")
        print(f"  Sessions : {len(sessions)}\n")
        _print_sessions(sessions)

        if not sessions:
            return

        if args.all_sessions:
            for s in sessions:
                _report_session(db, s)
        elif args.session is not None:
            matching = [s for s in sessions if s["session_id"] == args.session]
            if not matching:
                print(f"\n[ERROR] Session {args.session} not found.")
                sys.exit(1)
            _report_session(db, matching[0])
        else:
            # Default: most recent session
            _report_session(db, sessions[-1])


if __name__ == "__main__":
    main()

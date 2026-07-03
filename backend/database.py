"""
database.py — SQLite storage layer for CervixAI.
Handles creation of the analyses table and all CRUD used by main.py.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "cervixai.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            image_path TEXT NOT NULL,
            heatmap_path TEXT,
            cell_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            risk_level TEXT NOT NULL,
            probabilities TEXT NOT NULL,   -- JSON string of all class probabilities
            patient_name TEXT,
            notes TEXT,
            second_opinion_requested INTEGER DEFAULT 0,
            second_opinion_note TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()

    # Lightweight migration for databases created before these columns existed.
    existing_cols = {row["name"] for row in cur.execute("PRAGMA table_info(analyses)")}
    migrations = {
        "heatmap_path": "ALTER TABLE analyses ADD COLUMN heatmap_path TEXT",
        "second_opinion_requested": "ALTER TABLE analyses ADD COLUMN second_opinion_requested INTEGER DEFAULT 0",
        "second_opinion_note": "ALTER TABLE analyses ADD COLUMN second_opinion_note TEXT",
    }
    for col, stmt in migrations.items():
        if col not in existing_cols:
            cur.execute(stmt)
    conn.commit()
    conn.close()


def insert_analysis(
    scan_id: str,
    filename: str,
    image_path: str,
    cell_type: str,
    confidence: float,
    risk_level: str,
    probabilities: str,
    heatmap_path: Optional[str] = None,
    patient_name: Optional[str] = None,
    notes: Optional[str] = None,
) -> int:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO analyses
           (scan_id, filename, image_path, heatmap_path, cell_type, confidence, risk_level,
            probabilities, patient_name, notes, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            scan_id, filename, image_path, heatmap_path, cell_type, confidence, risk_level,
            probabilities, patient_name, notes, datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def set_second_opinion(scan_id: str, note: Optional[str] = None) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE analyses SET second_opinion_requested = 1, second_opinion_note = ? WHERE scan_id = ?",
        (note, scan_id),
    )
    conn.commit()
    updated = cur.rowcount > 0
    conn.close()
    return updated


def get_all_analyses(limit: int = 200):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM analyses ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_analysis_by_scan_id(scan_id: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM analyses WHERE scan_id = ?", (scan_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_analysis(scan_id: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM analyses WHERE scan_id = ?", (scan_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def get_dashboard_stats():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) as c FROM analyses")
    total = cur.fetchone()["c"]

    cur.execute("SELECT COUNT(*) as c FROM analyses WHERE risk_level = 'High Risk'")
    high = cur.fetchone()["c"]

    cur.execute("SELECT COUNT(*) as c FROM analyses WHERE risk_level = 'Normal'")
    normal = cur.fetchone()["c"]

    cur.execute("SELECT COUNT(*) as c FROM analyses WHERE risk_level = 'Low Risk'")
    low = cur.fetchone()["c"]

    cur.execute("SELECT COUNT(*) as c FROM analyses WHERE risk_level = 'Moderate Risk'")
    moderate = cur.fetchone()["c"]

    cur.execute("SELECT AVG(confidence) as a FROM analyses")
    avg_conf_row = cur.fetchone()["a"]
    avg_conf = round(avg_conf_row, 1) if avg_conf_row else 0.0

    conn.close()
    return {
        "total": total,
        "high_risk": high,
        "normal": normal,
        "low_risk": low,
        "moderate_risk": moderate,
        "avg_confidence": avg_conf,
    }
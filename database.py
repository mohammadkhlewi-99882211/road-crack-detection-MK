import json
import os
import sqlite3
from datetime import datetime


DB_PATH = "data/crack_analysis.db"


def get_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analysis_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            image_filename TEXT NOT NULL,
            image_path TEXT NOT NULL,
            result_image_path TEXT,
            num_cracks INTEGER DEFAULT 0,
            overall_severity TEXT,
            overall_confidence REAL DEFAULT 0,
            material_type TEXT,
            analysis_json TEXT,
            notes TEXT
        )
    """)
    conn.commit()
    return conn


def save_record(image_filename, image_path, result_image_path, num_cracks,
                overall_severity, overall_confidence, material_type, analysis_data, notes=""):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO analysis_records 
        (created_at, image_filename, image_path, result_image_path, num_cracks,
         overall_severity, overall_confidence, material_type, analysis_json, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        image_filename,
        image_path,
        result_image_path,
        num_cracks,
        overall_severity,
        overall_confidence,
        material_type,
        json.dumps(analysis_data, ensure_ascii=False),
        notes
    ))
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    return record_id


def get_all_records():
    conn = get_db()
    rows = conn.execute("SELECT * FROM analysis_records ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_record(record_id):
    conn = get_db()
    row = conn.execute("SELECT * FROM analysis_records WHERE id = ?", (record_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_record(record_id):
    conn = get_db()
    record = conn.execute("SELECT image_path, result_image_path FROM analysis_records WHERE id = ?", (record_id,)).fetchone()
    if record:
        for path_key in ["image_path", "result_image_path"]:
            path = record[path_key]
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
    conn.execute("DELETE FROM analysis_records WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()


def get_statistics():
    conn = get_db()
    stats = {}
    stats["total"] = conn.execute("SELECT COUNT(*) as c FROM analysis_records").fetchone()["c"]
    severity_rows = conn.execute(
        "SELECT overall_severity, COUNT(*) as c FROM analysis_records GROUP BY overall_severity"
    ).fetchall()
    stats["by_severity"] = {r["overall_severity"]: r["c"] for r in severity_rows}
    material_rows = conn.execute(
        "SELECT material_type, COUNT(*) as c FROM analysis_records GROUP BY material_type"
    ).fetchall()
    stats["by_material"] = {r["material_type"]: r["c"] for r in material_rows if r["material_type"]}
    avg_row = conn.execute(
        "SELECT AVG(overall_confidence) as avg_conf, AVG(num_cracks) as avg_cracks FROM analysis_records"
    ).fetchone()
    stats["avg_confidence"] = round(avg_row["avg_conf"] or 0, 1)
    stats["avg_cracks"] = round(avg_row["avg_cracks"] or 0, 1)
    recent = conn.execute(
        "SELECT * FROM analysis_records ORDER BY created_at DESC LIMIT 10"
    ).fetchall()
    stats["recent"] = [dict(r) for r in recent]
    conn.close()
    return stats

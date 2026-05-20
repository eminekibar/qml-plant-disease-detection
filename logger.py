"""
SQLite Event Logger
===================

Hedefler:
    - Sonuçları /data altında tutmak
    - Alanı kontrolsüz büyütmemek
    - Raspberry Pi tarafında bağımsız çalışmak

Bu nedenle iki katmanlı temizlik var:
    1. Gün bazlı temizlik
    2. Görsel klasörü maksimum boyut sınırı
"""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timedelta

import cv2


DEFAULT_LOG_DIR = "/data"
KEEP_DAYS = 7
MAX_IMAGE_DIR_MB = 1024


def _resolve_log_dir() -> str:
    env_path = os.environ.get("PLANT_PI_LOG_DIR")
    if env_path:
        return env_path
    if os.path.exists(DEFAULT_LOG_DIR):
        return DEFAULT_LOG_DIR
    return os.path.join(os.path.dirname(__file__), "local_data")


LOG_DIR = _resolve_log_dir()
DB_PATH = os.path.join(LOG_DIR, "plant_log.db")
IMG_DIR = os.path.join(LOG_DIR, "images")


def _ensure_dirs() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)


def _get_conn() -> sqlite3.Connection:
    _ensure_dirs()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            is_diseased INTEGER NOT NULL,
            image_path TEXT
        )
        """
    )
    conn.commit()
    return conn


_conn = None
_lock = threading.Lock()


def _conn_get() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = _get_conn()
    return _conn


def log_event(class_name: str, confidence: float, is_diseased: bool, frame=None) -> str | None:
    """
    Bir tanıma olayını kaydeder.

    frame verilirse JPEG olarak da saklanır. JPEG kalite değeri düşük ama yeterli
    tutuldu; amaç sınıflandırma sonucunu belgelemek, arşiv kalitesinde görsel değil.
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    image_path = None

    if frame is not None:
        filename = now.strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
        image_path = os.path.join(IMG_DIR, filename)
        cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    with _lock:
        conn = _conn_get()
        conn.execute(
            "INSERT INTO events (timestamp, class_name, confidence, is_diseased, image_path) VALUES (?,?,?,?,?)",
            (timestamp, class_name, confidence, int(is_diseased), image_path),
        )
        conn.commit()

    return image_path


def purge_old_records(keep_days: int = KEEP_DAYS) -> int:
    cutoff = (datetime.now() - timedelta(days=keep_days)).strftime("%Y-%m-%d %H:%M:%S")

    with _lock:
        conn = _conn_get()
        rows = conn.execute(
            "SELECT image_path FROM events WHERE timestamp < ?",
            (cutoff,),
        ).fetchall()

        for (path,) in rows:
            if path and os.path.exists(path):
                os.remove(path)

        conn.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
        conn.commit()

    return len(rows)


def purge_storage_budget(max_image_dir_mb: int = MAX_IMAGE_DIR_MB) -> int:
    """
    Görsel klasörü boyutunu kontrol altında tutar.

    Gün bazlı temizlik çoğu zaman yeterlidir; fakat yoğun demo sırasında aynı gün
    içinde binlerce kayıt oluşursa ek güvenlik olarak boyut sınırı uygulanır.
    """
    budget_bytes = max_image_dir_mb * 1024 * 1024

    with _lock:
        conn = _conn_get()
        rows = conn.execute(
            "SELECT id, image_path FROM events WHERE image_path IS NOT NULL ORDER BY id ASC"
        ).fetchall()

        total_size = 0
        files = []
        for event_id, image_path in rows:
            if image_path and os.path.exists(image_path):
                size = os.path.getsize(image_path)
                total_size += size
                files.append((event_id, image_path, size))

        deleted = 0
        while total_size > budget_bytes and files:
            event_id, image_path, size = files.pop(0)
            if os.path.exists(image_path):
                os.remove(image_path)
            conn.execute("UPDATE events SET image_path = NULL WHERE id = ?", (event_id,))
            total_size -= size
            deleted += 1

        conn.commit()

    return deleted


def recent_events(limit: int = 20) -> list[dict]:
    with _lock:
        conn = _conn_get()
        rows = conn.execute(
            "SELECT timestamp, class_name, confidence, is_diseased, image_path "
            "FROM events ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    return [
        {
            "timestamp": row[0],
            "class_name": row[1],
            "confidence": round(row[2], 3),
            "is_diseased": bool(row[3]),
            "image_path": row[4],
        }
        for row in rows
    ]


def close() -> None:
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None

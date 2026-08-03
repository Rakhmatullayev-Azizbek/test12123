"""
metadata.db (SQLite) - qaysi hujjatlar scrape va index qilinganini kuzatadi.

Bu modul scraper va indexer uchun umumiy: ikkala servis ham ./data volume
orqali bir xil faylga murojaat qiladi. Shuning uchun WAL rejimi va
busy_timeout yoqilgan - parallel yozishda "database is locked" bo'lmasligi uchun.

Jadval sxemasi:
    document_id   TEXT PRIMARY KEY
    content_hash  TEXT      -- HTML mazmunining hash'i (o'zgarishni aniqlash uchun)
    status        TEXT      -- 'new' | 'updated' | 'unchanged' | 'indexed' | 'failed'
    scraped_at    TEXT
    indexed_at    TEXT      -- NULL bo'lsa: indekslash kutilmoqda
    chunk_count   INTEGER   -- Qdrant'ga yozilgan chunk soni
    last_error    TEXT      -- oxirgi indekslash xatosi (bo'lsa)
"""
import hashlib
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("shared.metadata_db")

# Indekslashga tayyor holatlar
PENDING_STATUSES = ("new", "updated", "failed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect(db_path: str):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    """Jadvalni yaratadi va eski bazalarga yangi ustunlarni qo'shadi (migratsiya)."""
    with _connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id   TEXT PRIMARY KEY,
                content_hash  TEXT NOT NULL,
                status        TEXT NOT NULL,
                scraped_at    TEXT NOT NULL,
                indexed_at    TEXT
            )
        """)
        existing = {row[1] for row in conn.execute("PRAGMA table_info(documents)")}
        for column, ddl in (
            ("chunk_count", "ALTER TABLE documents ADD COLUMN chunk_count INTEGER"),
            ("last_error", "ALTER TABLE documents ADD COLUMN last_error TEXT"),
        ):
            if column not in existing:
                conn.execute(ddl)
                logger.info(f"metadata.db migratsiya: '{column}' ustuni qo'shildi")


def _hash_content(html: str) -> str:
    return hashlib.sha256(html.encode("utf-8")).hexdigest()


def check_and_update(db_path: str, document_id: str, html: str) -> str:
    """
    Hujjatni tekshiradi, holatini yangilaydi va status qaytaradi:
    'new' | 'updated' | 'unchanged'

    'new'/'updated' bo'lsa indexed_at NULL ga qaytariladi - indexer uni
    qayta indekslashi kerakligini shu orqali biladi.
    """
    content_hash = _hash_content(html)

    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT content_hash FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()

        if row is None:
            status = "new"
        elif row[0] != content_hash:
            status = "updated"
        else:
            return "unchanged"

        conn.execute(
            """
            INSERT INTO documents
                (document_id, content_hash, status, scraped_at, indexed_at, chunk_count, last_error)
            VALUES (?, ?, ?, ?, NULL, NULL, NULL)
            ON CONFLICT(document_id) DO UPDATE SET
                content_hash = excluded.content_hash,
                status       = excluded.status,
                scraped_at   = excluded.scraped_at,
                indexed_at   = NULL,
                chunk_count  = NULL,
                last_error   = NULL
            """,
            (document_id, content_hash, status, _now()),
        )
        return status


def list_pending(db_path: str, include_all: bool = False) -> list[str]:
    """
    Indekslashni kutayotgan hujjat ID'larini qaytaradi.

    include_all=True bo'lsa - indexed_at qiymatidan qat'i nazar hammasi
    (REINDEX_ALL uchun).
    """
    with _connect(db_path) as conn:
        if include_all:
            rows = conn.execute(
                "SELECT document_id FROM documents ORDER BY document_id"
            ).fetchall()
        else:
            placeholders = ",".join("?" * len(PENDING_STATUSES))
            rows = conn.execute(
                f"""
                SELECT document_id FROM documents
                WHERE indexed_at IS NULL OR status IN ({placeholders})
                ORDER BY document_id
                """,
                PENDING_STATUSES,
            ).fetchall()
    return [r[0] for r in rows]


def mark_indexed(db_path: str, document_id: str, chunk_count: int) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE documents
            SET status = 'indexed', indexed_at = ?, chunk_count = ?, last_error = NULL
            WHERE document_id = ?
            """,
            (_now(), chunk_count, document_id),
        )


def mark_failed(db_path: str, document_id: str, error: str) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE documents
            SET status = 'failed', indexed_at = NULL, last_error = ?
            WHERE document_id = ?
            """,
            (error[:1000], document_id),
        )


def stats(db_path: str) -> dict:
    """Umumiy holat: /health va loglar uchun."""
    with _connect(db_path) as conn:
        total, indexed, chunks = conn.execute(
            """
            SELECT COUNT(*),
                   COUNT(indexed_at),
                   COALESCE(SUM(chunk_count), 0)
            FROM documents
            """
        ).fetchone()
    return {
        "documents_total": total,
        "documents_indexed": indexed,
        "documents_pending": total - indexed,
        "chunks_indexed": chunks,
    }

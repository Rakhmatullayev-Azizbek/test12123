"""
Cookie orqali kelgan har bir brauzer/qurilma uchun user_id (UUID) asosida
sessiya va chat tarixini PostgreSQL'da saqlaydi.

Ulanishlar pool orqali boshqariladi: ilgari har bir so'rovda 3 ta yangi
ulanish ochilardi (touch_session + log_message + o'qish), bu yuklamada
Postgres'ni tez charchatardi.

Jadval sxemasi:
    sessions:
        user_id         TEXT PRIMARY KEY   -- cookie qiymati (UUID)
        first_seen_at   TIMESTAMPTZ        -- birinchi marta ochilgan vaqt
        last_seen_at    TIMESTAMPTZ        -- oxirgi faollik vaqti
        message_count   INTEGER            -- jami savollar soni

    messages:
        id              SERIAL PRIMARY KEY
        user_id         TEXT               -- sessions.user_id ga bog'liq
        question        TEXT
        answer          TEXT
        created_at      TIMESTAMPTZ
"""
import logging
import threading
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from psycopg2 import pool as pg_pool

from shared.config import settings

logger = logging.getLogger("app.session_store")

_pool: pg_pool.ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()


def _get_pool() -> pg_pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = pg_pool.ThreadedConnectionPool(
                    minconn=settings.POSTGRES_POOL_MIN,
                    maxconn=settings.POSTGRES_POOL_MAX,
                    host=settings.POSTGRES_HOST,
                    port=settings.POSTGRES_PORT,
                    dbname=settings.POSTGRES_DB,
                    user=settings.POSTGRES_USER,
                    password=settings.POSTGRES_PASSWORD,
                )
                logger.info(
                    f"PostgreSQL pool yaratildi "
                    f"({settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB})"
                )
    return _pool


@contextmanager
def _cursor(dict_rows: bool = False):
    """Pool'dan ulanish oladi, tranzaksiyani yakunlaydi va ulanishni qaytaradi."""
    connection_pool = _get_pool()
    conn = connection_pool.getconn()
    factory = psycopg2.extras.RealDictCursor if dict_rows else None
    try:
        with conn:
            with conn.cursor(cursor_factory=factory) as cur:
                yield cur
    except Exception:
        conn.rollback()
        raise
    finally:
        connection_pool.putconn(conn)


def close_pool() -> None:
    """Ilova to'xtaganda ulanishlarni yopadi."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
        logger.info("PostgreSQL pool yopildi")


def is_healthy() -> bool:
    """PostgreSQL ma'lumotlar bazasiga ulanishni tekshirish (Health check uchun)."""
    try:
        with _cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:
        return False


def init_db() -> None:
    with _cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                user_id         TEXT PRIMARY KEY,
                first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_seen_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                message_count   INTEGER NOT NULL DEFAULT 0
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id              SERIAL PRIMARY KEY,
                user_id         TEXT NOT NULL,
                question        TEXT NOT NULL,
                answer          TEXT NOT NULL,
                created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        # Tarixni user bo'yicha o'qish uchun indeks
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_user_id_id
            ON messages (user_id, id DESC)
        """)


def touch_session(user_id: str) -> None:
    """
    Sessiyani yangilaydi: agar user_id yangi bo'lsa - sessions'ga qo'shadi
    (first_seen_at hozirgi vaqt bilan), aks holda last_seen_at'ni yangilaydi.
    """
    with _cursor() as cur:
        cur.execute(
            """
            INSERT INTO sessions (user_id, first_seen_at, last_seen_at, message_count)
            VALUES (%s, now(), now(), 0)
            ON CONFLICT (user_id) DO UPDATE SET last_seen_at = now()
            """,
            (user_id,),
        )


def log_message(user_id: str, question: str, answer: str) -> None:
    """Savol-javobni tarixga yozadi va sessiyaning message_count'ini oshiradi."""
    with _cursor() as cur:
        cur.execute(
            "INSERT INTO messages (user_id, question, answer) VALUES (%s, %s, %s)",
            (user_id, question, answer),
        )
        cur.execute(
            """
            UPDATE sessions
            SET message_count = message_count + 1, last_seen_at = now()
            WHERE user_id = %s
            """,
            (user_id,),
        )


def get_session_info(user_id: str) -> dict | None:
    """Bitta user uchun sessiya ma'lumotini qaytaradi (first_seen_at, last_seen_at, message_count)."""
    with _cursor(dict_rows=True) as cur:
        cur.execute(
            "SELECT user_id, first_seen_at, last_seen_at, message_count "
            "FROM sessions WHERE user_id = %s",
            (user_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def get_message_history(user_id: str, limit: int = 50) -> list[dict]:
    """Bitta user uchun oxirgi savol-javoblar tarixini qaytaradi (eng yangisi oxirida)."""
    with _cursor(dict_rows=True) as cur:
        cur.execute(
            """
            SELECT question, answer, created_at FROM messages
            WHERE user_id = %s
            ORDER BY id DESC
            LIMIT %s
            """,
            (user_id, limit),
        )
        rows = cur.fetchall()
    return [dict(r) for r in reversed(rows)]


def clear_history(user_id: str) -> int:
    """Sessiya tarixini tozalaydi. Qaytaradi: o'chirilgan xabarlar soni."""
    with _cursor() as cur:
        cur.execute("DELETE FROM messages WHERE user_id = %s", (user_id,))
        deleted = cur.rowcount
        cur.execute(
            "UPDATE sessions SET message_count = 0 WHERE user_id = %s", (user_id,)
        )
    return deleted


def get_all_sessions() -> list[dict]:
    """Barcha userlarning faolligini ko'rish uchun (admin/statistika maqsadida)."""
    with _cursor(dict_rows=True) as cur:
        cur.execute(
            "SELECT user_id, first_seen_at, last_seen_at, message_count "
            "FROM sessions ORDER BY last_seen_at DESC"
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]

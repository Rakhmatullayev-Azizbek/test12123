"""
Umumiy konfiguratsiya. Barcha servislar (scraper, indexer, rag_backend)
shu modulni import qilib, .env dan sozlamalarni oladi.
"""
import os


def _get_env(key: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(key, default)
    if required and not value:
        raise RuntimeError(
            f"Muhim environment variable topilmadi: {key}. "
            f".env faylini tekshiring (.env.example asosida yarating)."
        )
    return value


def _get_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"{key} butun son bo'lishi kerak, olindi: {raw!r}")


class Settings:
    # ---------- Qdrant ----------
    QDRANT_HOST: str = _get_env("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = _get_int("QDRANT_PORT_INTERNAL", 6565)
    QDRANT_COLLECTION: str = _get_env("QDRANT_COLLECTION", "tr_legal_docs_v1")

    # ---------- vLLM ----------
    # DIQQAT: bu chatbot endi EISVO OCR loyihasidagi vLLM serveriga ulanadi —
    # Qwen/Qwen3-14B-AWQ, `vllm` servisi, docker tarmog'ida http://vllm:8000/v1.
    # (Ilgari alohida gemma/brb_bank_model 8050-portда ishlatilardi.)
    VLLM_API_URL: str = _get_env("VLLM_API_URL", required=True)
    VLLM_MODEL_NAME: str = _get_env("VLLM_MODEL_NAME", "Qwen/Qwen3-14B-AWQ")
    VLLM_TIMEOUT: float = float(_get_env("VLLM_TIMEOUT", "120"))
    VLLM_MAX_TOKENS: int = _get_int("VLLM_MAX_TOKENS", 1024)
    VLLM_TEMPERATURE: float = float(_get_env("VLLM_TEMPERATURE", "0.2"))
    # Gemma "system" rolini qabul qilmaydi, Qwen3 esa QABUL QILADI — shuning uchun
    # default true. (false bo'lsa ko'rsatma birinchi user xabariga qo'shiladi.)
    LLM_SUPPORTS_SYSTEM_ROLE: bool = _get_env(
        "LLM_SUPPORTS_SYSTEM_ROLE", "true"
    ).lower() in ("1", "true", "yes")
    # Qwen3 "thinking" rejimida javob oldidan <think>...</think> qismini yozadi —
    # RAG javobiga bu keraksiz shovqin. false bo'lsa vLLM'ga chat_template_kwargs
    # orqali enable_thinking=false yuboriladi (gemma'da bu parametr bo'lmagan).
    LLM_ENABLE_THINKING: bool = _get_env(
        "LLM_ENABLE_THINKING", "false"
    ).lower() in ("1", "true", "yes")
    # Kontekst uchun ajratiladigan maksimal belgi soni (model oynasiga sig'ishi uchun).
    # Qwen3-14B-AWQ bu loyihada --max-model-len 20480 token; ~3 belgi/token deb
    # hisoblab, savol + tarix + javob (max_tokens) uchun zaxira qoldiramiz.
    LLM_CONTEXT_CHAR_BUDGET: int = _get_int("LLM_CONTEXT_CHAR_BUDGET", 18000)

    # ---------- Embedding ----------
    EMBEDDING_MODEL: str = _get_env("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    EMBEDDING_DIM: int = _get_int("EMBEDDING_DIM", 1024)
    EMBEDDING_BATCH_SIZE: int = _get_int("EMBEDDING_BATCH_SIZE", 16)
    # e5-large 512 token bilan cheklangan. Undan uzun chunk jimgina qirqiladi,
    # shuning uchun indekslashdan oldin o'zimiz bo'laklarga ajratamiz.
    MAX_CHUNK_CHARS: int = _get_int("MAX_CHUNK_CHARS", 1200)
    CHUNK_OVERLAP_CHARS: int = _get_int("CHUNK_OVERLAP_CHARS", 150)
    # Bitta chunk uchun saqlanadigan context_text chegarasi (LLM'ga beriladigan qism).
    # Butun bobni tashlash o'rniga band atrofidagi qo'shnilar bilan cheklanadi.
    MAX_CONTEXT_CHARS: int = _get_int("MAX_CONTEXT_CHARS", 3000)

    # ---------- Scraper ----------
    LEX_UZ_BASE_URL: str = _get_env("LEX_UZ_BASE_URL", "https://lex.uz")
    SCRAPE_INTERVAL_HOURS: int = _get_int("SCRAPE_INTERVAL_HOURS", 12)
    DOCUMENT_LIST_PATH: str = _get_env("DOCUMENT_LIST_PATH", "/app/document_urls.txt")

    # ---------- Indexer ----------
    INDEX_INTERVAL_MINUTES: int = _get_int("INDEX_INTERVAL_MINUTES", 30)
    # true bo'lsa - ishga tushganda hamma hujjat qayta indekslanadi (indexed_at e'tiborsiz)
    REINDEX_ALL: bool = _get_env("REINDEX_ALL", "false").lower() in ("1", "true", "yes")

    # ---------- Backend ----------
    ALLOWED_ORIGINS: list[str] = [
        o.strip() for o in _get_env("ALLOWED_ORIGINS", "*").split(",") if o.strip()
    ]
    RETRIEVAL_SCORE_THRESHOLD: float = float(_get_env("RETRIEVAL_SCORE_THRESHOLD", "0.80"))
    HISTORY_TURNS_IN_PROMPT: int = _get_int("HISTORY_TURNS_IN_PROMPT", 3)

    # Kuzatuv savollarini ("Angliyachi?") tarix asosida mustaqil savolga
    # aylantirish. Bu bo'lmasa qidiruv xom savolni vektorlashtiradi va
    # hech narsa topolmaydi - LLM'ga kontekst umuman yetib bormaydi.
    QUERY_REWRITE_ENABLED: bool = _get_env("QUERY_REWRITE_ENABLED", "true").lower() in (
        "1", "true", "yes",
    )
    QUERY_REWRITE_MAX_TOKENS: int = _get_int("QUERY_REWRITE_MAX_TOKENS", 120)
    # Qayta yozilgan savol shundan uzun bo'lsa - model ko'rsatmaga amal qilmagan,
    # asl savolga qaytamiz
    QUERY_REWRITE_MAX_CHARS: int = _get_int("QUERY_REWRITE_MAX_CHARS", 300)

    # ---------- Fayl yo'llari (container ichida) ----------
    DATA_DIR: str = _get_env("DATA_DIR", "/app/data")
    RAW_DIR: str = f"{DATA_DIR}/raw"
    METADATA_DB_PATH: str = f"{DATA_DIR}/metadata.db"

    # ---------- PostgreSQL (sessiya/chat tarixi uchun) ----------
    POSTGRES_HOST: str = _get_env("POSTGRES_HOST", "postgres")
    # DIQQAT: .env dagi POSTGRES_PORT - host mashinasidagi port (compose mapping uchun).
    # Konteyner tarmog'i ichida Postgres doim 5432 da tinglaydi.
    POSTGRES_PORT: int = _get_int("POSTGRES_PORT_INTERNAL", 5432)
    POSTGRES_DB: str = _get_env("POSTGRES_DB", "tr_chatbot")
    POSTGRES_USER: str = _get_env("POSTGRES_USER", "tr_chatbot")
    POSTGRES_PASSWORD: str = _get_env("POSTGRES_PASSWORD", "tr_chatbot")
    POSTGRES_POOL_MIN: int = _get_int("POSTGRES_POOL_MIN", 1)
    POSTGRES_POOL_MAX: int = _get_int("POSTGRES_POOL_MAX", 10)

    # ---------- Cookie ----------
    SESSION_COOKIE_NAME: str = _get_env("SESSION_COOKIE_NAME", "tr_chatbot_uid")
    SESSION_COOKIE_MAX_AGE_DAYS: int = _get_int("SESSION_COOKIE_MAX_AGE_DAYS", 365)
    # Frontend boshqa domenda bo'lsa: SAMESITE=none va SECURE=true (HTTPS talab qiladi)
    SESSION_COOKIE_SAMESITE: str = _get_env("SESSION_COOKIE_SAMESITE", "lax")
    SESSION_COOKIE_SECURE: bool = _get_env("SESSION_COOKIE_SECURE", "false").lower() in (
        "1", "true", "yes",
    )


settings = Settings()

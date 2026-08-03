"""
Holat va statistika endpointlari.
"""
import logging

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from app.models.schemas import HealthResponse, StatsResponse
from app.services import qdrant_client, vllm_client
from app.services.session_store import is_healthy as postgres_is_healthy
from shared import metadata_db
from shared.config import settings

logger = logging.getLogger("app.health")

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Servislar holati",
    description=(
        "Qdrant, PostgreSQL va LLM bilan aloqani tekshiradi. "
        "`status` = `ok` faqat barcha muhim komponentlar ishlaganda "
        "(kolleksiya mavjudligi ham talab qilinadi)."
    ),
)
async def health_check() -> HealthResponse:
    qdrant_ok = await run_in_threadpool(qdrant_client.is_healthy)
    postgres_ok = await run_in_threadpool(postgres_is_healthy)
    collection = await run_in_threadpool(qdrant_client.collection_exists)
    chunks = await run_in_threadpool(qdrant_client.count_points) if collection else 0
    llm_ok = await vllm_client.check_reachable()

    healthy = qdrant_ok and postgres_ok and llm_ok and collection and chunks > 0
    return HealthResponse(
        status="ok" if healthy else "degraded",
        qdrant_connected=qdrant_ok,
        postgres_connected=postgres_ok,
        llm_reachable=llm_ok,
        collection_exists=collection,
        indexed_chunks=chunks,
    )


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Indekslash statistikasi",
    description=(
        "Nechta hujjat scrape/index qilingani va Qdrant'da nechta bo'lak "
        "borligini ko'rsatadi. Indexer ishlayotganini tekshirish uchun."
    ),
)
async def stats() -> StatsResponse:
    chunks = await run_in_threadpool(qdrant_client.count_points)

    # metadata.db - scraper/indexer bilan umumiy volume'da. Backend'da bo'lmasa
    # (masalan volume ulanmagan) statistika nolga tushadi, xato bermaydi.
    doc_stats = {"documents_total": 0, "documents_indexed": 0, "documents_pending": 0}
    try:
        doc_stats = await run_in_threadpool(metadata_db.stats, settings.METADATA_DB_PATH)
    except Exception as e:
        logger.warning(f"metadata.db o'qilmadi: {e}")

    return StatsResponse(
        collection=settings.QDRANT_COLLECTION,
        indexed_chunks=chunks,
        documents_total=doc_stats["documents_total"],
        documents_indexed=doc_stats["documents_indexed"],
        documents_pending=doc_stats["documents_pending"],
        embedding_model=settings.EMBEDDING_MODEL,
        llm_model=settings.VLLM_MODEL_NAME,
    )

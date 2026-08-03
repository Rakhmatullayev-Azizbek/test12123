"""
Qdrant bilan ishlash (faqat o'qish - yozishni indexer bajaradi).
"""
import logging

from qdrant_client import QdrantClient, models

from shared.config import settings

logger = logging.getLogger("app.qdrant")

_client: QdrantClient | None = None


class CollectionMissingError(RuntimeError):
    """Kolleksiya hali yaratilmagan - indexer ishga tushmagan."""


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=30,
        )
    return _client


def is_healthy() -> bool:
    try:
        get_client().get_collections()
        return True
    except Exception:
        return False


def collection_exists() -> bool:
    try:
        return get_client().collection_exists(settings.QDRANT_COLLECTION)
    except Exception:
        return False


def count_points() -> int:
    try:
        return get_client().count(
            collection_name=settings.QDRANT_COLLECTION, exact=True
        ).count
    except Exception:
        return 0


def search_chunks(
    query_vector: list[float],
    top_k: int = 5,
    section_type: str | None = None,
    score_threshold: float | None = None,
) -> list[dict]:
    """
    Qidiruv embed_text vektori bo'yicha amalga oshiriladi, lekin LLM'ga
    context_text (kengaytirilgan matn) yuboriladi - hybrid chunking
    yondashuvi shu yerda ishlatiladi.

    Qaytaradi: payload dict ro'yxati, har birida "score" maydoni qo'shilgan.
    """
    if not collection_exists():
        raise CollectionMissingError(
            f"Qdrant kolleksiyasi '{settings.QDRANT_COLLECTION}' topilmadi. "
            f"Indexer ishga tushganini tekshiring: docker compose logs indexer"
        )

    query_filter = None
    if section_type:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="section_type", match=models.MatchValue(value=section_type)
                )
            ]
        )

    response = get_client().query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
        score_threshold=score_threshold,
        with_payload=True,
    )

    results: list[dict] = []
    for point in response.points:
        payload = dict(point.payload or {})
        payload["score"] = point.score
        results.append(payload)
    return results

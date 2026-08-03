"""
Qdrant kolleksiyasini yaratish va chunk'larni yozish.

Muhim jihatlar:
    - Kolleksiya avtomatik yaratiladi (oldin hech qachon yaratilmagan edi -
      shu sababli qidiruv 404 qaytarardi)
    - document_id bo'yicha payload indeksi - hujjatni o'chirish/filtrlash uchun
    - Hujjat qayta indekslanganda avval uning barcha eski chunk'lari o'chiriladi,
      shunda yangilangan hujjatdan qolgan "arvoh" chunk'lar to'planmaydi
    - chunk_id deterministik (uuid5) - qayta yozishda dublikat paydo bo'lmaydi
"""
import logging

from qdrant_client import QdrantClient, models

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from shared.config import settings

logger = logging.getLogger("indexer.qdrant_writer")

_client: QdrantClient | None = None

# Payload maydonlari uchun indekslar (filtrlash tezligi uchun)
_INDEXED_PAYLOAD_FIELDS = {
    "document_id": models.PayloadSchemaType.KEYWORD,
    "section_type": models.PayloadSchemaType.KEYWORD,
    "kind": models.PayloadSchemaType.KEYWORD,
}


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=120,
        )
    return _client


def ensure_collection() -> None:
    """Kolleksiya mavjudligini kafolatlaydi (bo'lmasa yaratadi)."""
    client = get_client()
    name = settings.QDRANT_COLLECTION

    if client.collection_exists(name):
        info = client.get_collection(name)
        existing_dim = info.config.params.vectors.size
        if existing_dim != settings.EMBEDDING_DIM:
            raise RuntimeError(
                f"Qdrant kolleksiyasi '{name}' {existing_dim} o'lchamli vektor bilan "
                f"yaratilgan, lekin sozlamada EMBEDDING_DIM={settings.EMBEDDING_DIM}. "
                f"Kolleksiyani o'chirib qayta yaratish yoki QDRANT_COLLECTION nomini "
                f"o'zgartirish kerak."
            )
    else:
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=settings.EMBEDDING_DIM,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(
            f"Qdrant kolleksiyasi yaratildi: {name} "
            f"(dim={settings.EMBEDDING_DIM}, distance=COSINE)"
        )

    # Payload indekslari - allaqachon bo'lsa xatolik beradi, shuning uchun yumshoq o'tamiz
    for field, schema in _INDEXED_PAYLOAD_FIELDS.items():
        try:
            client.create_payload_index(
                collection_name=name, field_name=field, field_schema=schema
            )
        except Exception:
            pass


def delete_document(document_id: str) -> None:
    """Bitta hujjatga tegishli barcha chunk'larni o'chiradi (qayta indekslashdan oldin)."""
    get_client().delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    )
                ]
            )
        ),
        wait=True,
    )


def upsert_chunks(chunks: list[dict], vectors: list[list[float]]) -> int:
    """
    Chunk'larni vektorlari bilan Qdrant'ga yozadi.

    chunks[i] va vectors[i] bir-biriga mos bo'lishi shart.
    """
    if not chunks:
        return 0
    if len(chunks) != len(vectors):
        raise ValueError(
            f"chunk va vektor soni mos emas: {len(chunks)} != {len(vectors)}"
        )

    points = [
        models.PointStruct(
            id=chunk["chunk_id"],
            vector=vector,
            payload={k: v for k, v in chunk.items() if k != "chunk_id"},
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    get_client().upsert(
        collection_name=settings.QDRANT_COLLECTION,
        points=points,
        wait=True,
    )
    return len(points)


def count_points() -> int:
    try:
        return get_client().count(
            collection_name=settings.QDRANT_COLLECTION, exact=True
        ).count
    except Exception:
        return 0

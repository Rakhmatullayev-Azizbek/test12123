"""
multilingual-e5-large modeli orqali matnlarni vektorlashtirish.

E5 modellari uchun muhim qoida: matn oldiga "passage:" (saqlashda) yoki
"query:" (qidiruvda) prefiksini qo'shish kerak - aks holda natija sifati
sezilarli pasayadi.

Model 512 token bilan cheklangan: undan uzun matn jimgina qirqiladi.
Shuning uchun indekslashdan oldin chunk'lar shared.chunking orqali
MAX_CHUNK_CHARS gacha bo'laklanadi.
"""
import logging
import threading

from sentence_transformers import SentenceTransformer

from .config import settings

logger = logging.getLogger("shared.embedder")

_model: SentenceTransformer | None = None
_lock = threading.Lock()


def _pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_model() -> SentenceTransformer:
    """Modelni bir marta yuklaydi (thread-safe, lazy singleton)."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                device = _pick_device()
                logger.info(
                    f"Embedding modeli yuklanmoqda: {settings.EMBEDDING_MODEL} (device={device})"
                )
                _model = SentenceTransformer(settings.EMBEDDING_MODEL, device=device)
                logger.info(
                    f"Model yuklandi. max_seq_length={_model.max_seq_length}, "
                    f"vector_dim={_model.get_sentence_embedding_dimension()}"
                )
    return _model


def warmup() -> None:
    """Modelni oldindan yuklab, birinchi so'rovdagi kechikishni yo'q qiladi."""
    get_model().encode(["query: warmup"], normalize_embeddings=True)


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Qdrant'ga saqlash uchun (chunk matnlari). Batch bo'yicha ishlaydi."""
    if not texts:
        return []
    prefixed = [f"passage: {t}" for t in texts]
    vectors = get_model().encode(
        prefixed,
        normalize_embeddings=True,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Foydalanuvchi savolini vektorlashtirish uchun."""
    return get_model().encode(
        [f"query: {text}"], normalize_embeddings=True, show_progress_bar=False
    )[0].tolist()

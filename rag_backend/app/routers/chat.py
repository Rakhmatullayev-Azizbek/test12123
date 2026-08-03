"""
Asosiy RAG endpointlari: /chat (savol-javob) va /search (LLM'siz qidiruv).
"""
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status
from starlette.concurrency import run_in_threadpool

from app.deps import Session, resolve_session
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    SourceChunk,
)
from app.services import session_store
from app.services.qdrant_client import CollectionMissingError, search_chunks
from app.services.vllm_client import LLMUnavailableError, generate_answer, rewrite_query
from shared.config import settings
from shared.embedder import embed_query

logger = logging.getLogger("app.chat")

router = APIRouter()


def _unique_sources(chunks: list[dict]) -> list[SourceChunk]:
    """
    Manbalarni takrorsiz qaytaradi.

    Bir hujjatning bir bandidan bir nechta chunk kelishi mumkin (uzun band
    bo'laklangan) - foydalanuvchiga ularni takror ko'rsatish keraksiz.
    """
    seen: set[tuple] = set()
    sources: list[SourceChunk] = []
    for chunk in chunks:
        key = (
            chunk.get("document_id"),
            chunk.get("band_number"),
            chunk.get("element_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            SourceChunk(
                document_id=str(chunk.get("document_id", "")),
                document_title=str(chunk.get("document_title", "")),
                section_type=chunk.get("section_type"),
                chapter=chunk.get("chapter"),
                band_number=chunk.get("band_number"),
                score=round(chunk["score"], 4) if chunk.get("score") is not None else None,
                source_url=chunk.get("source_url"),
            )
        )
    return sources


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Qonunchilik bo'yicha savol berish",
    description=(
        "Savolni vektorlashtiradi, Qdrant'dan eng mos hujjat bo'laklarini topadi "
        "va ularni kontekst sifatida LLM'ga berib javob oladi.\n\n"
        "Sessiya cookie (`tr_chatbot_uid`) orqali avtomatik boshqariladi: birinchi "
        "so'rovda yaratiladi, keyingi so'rovlarda suhbat tarixi hisobga olinadi."
    ),
    responses={
        503: {"model": ErrorResponse, "description": "Qdrant kolleksiyasi yo'q yoki LLM ishlamayapti"},
    },
)
async def chat(
    request: ChatRequest,
    session: Session = Depends(resolve_session),
) -> ChatResponse:
    started = time.perf_counter()

    # Bloklovchi chaqiruvlar (model inferensi, Qdrant, Postgres) event loop'ni
    # to'sib qo'ymasligi uchun threadpool'da bajariladi.
    await run_in_threadpool(session_store.touch_session, session.user_id)

    history: list[dict] = []
    if request.use_history and not session.is_new:
        history = await run_in_threadpool(
            session_store.get_message_history,
            session.user_id,
            settings.HISTORY_TURNS_IN_PROMPT,
        )

    # Kuzatuv savolini ("Angliyachi?") qidirishdan OLDIN mustaqil savolga
    # aylantiramiz - aks holda vektor qidiruv hech narsa topolmaydi.
    search_question = await rewrite_query(request.question, history)

    query_vector = await run_in_threadpool(embed_query, search_question)

    try:
        chunks = await run_in_threadpool(
            search_chunks,
            query_vector,
            request.top_k,
            None,
            settings.RETRIEVAL_SCORE_THRESHOLD,
        )
    except CollectionMissingError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from e
    except Exception as e:
        logger.exception("Qdrant qidiruvida xatolik")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"Qidiruv bazasi ishlamayapti: {e}"
        ) from e

    # Javob generatsiyasiga ham to'ldirilgan savolni beramiz - kontekst aynan
    # shu savol bo'yicha topilgan, shuning uchun mos tushadi.
    try:
        answer = await generate_answer(search_question, chunks, history)
    except LLMUnavailableError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from e

    # Tarixga yozish javobni qaytarishga to'sqinlik qilmasin
    try:
        await run_in_threadpool(
            session_store.log_message, session.user_id, request.question, answer
        )
    except Exception:
        logger.exception("Chat tarixini yozishda xatolik (javob baribir qaytariladi)")

    return ChatResponse(
        answer=answer,
        sources=_unique_sources(chunks),
        session_id=session.user_id,
        retrieved_count=len(chunks),
        rewritten_question=search_question if search_question != request.question else None,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Faqat vektor qidiruv (LLM'siz)",
    description=(
        "LLM'ni chaqirmasdan, Qdrant'dan qaytgan xom natijalarni ko'rsatadi. "
        "Indekslash sifatini tekshirish va nosozlikni aniqlash uchun qulay."
    ),
    responses={503: {"model": ErrorResponse, "description": "Qdrant kolleksiyasi yo'q"}},
)
async def search(request: SearchRequest) -> SearchResponse:
    query_vector = await run_in_threadpool(embed_query, request.query)

    try:
        chunks = await run_in_threadpool(
            search_chunks,
            query_vector,
            request.top_k,
            request.section_type,
            None,  # debug uchun chegara qo'ymaymiz - hammasi ko'rinsin
        )
    except CollectionMissingError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from e

    return SearchResponse(
        query=request.query,
        hits=[
            SearchHit(
                score=round(c.get("score") or 0.0, 4),
                document_id=str(c.get("document_id", "")),
                document_title=str(c.get("document_title", "")),
                section_type=c.get("section_type"),
                chapter=c.get("chapter"),
                band_number=c.get("band_number"),
                source_url=c.get("source_url"),
                embed_text=str(c.get("embed_text", "")),
            )
            for c in chunks
        ],
    )

"""
Sessiya va suhbat tarixi endpointlari.
"""
import logging

from fastapi import APIRouter, Depends, Query
from starlette.concurrency import run_in_threadpool

from app.deps import Session, resolve_session
from app.models.schemas import HistoryItem, HistoryResponse
from app.services import session_store

logger = logging.getLogger("app.session")

router = APIRouter()


def _iso(value) -> str | None:
    return value.isoformat() if value is not None else None


@router.get(
    "/history",
    response_model=HistoryResponse,
    summary="Joriy sessiyaning savol-javob tarixi",
    description=(
        "Cookie (`tr_chatbot_uid`) orqali aniqlangan sessiyaning suhbat tarixini "
        "qaytaradi (eng yangisi ro'yxat oxirida)."
    ),
)
async def get_history(
    session: Session = Depends(resolve_session),
    limit: int = Query(50, ge=1, le=200, description="Qaytariladigan xabarlar soni."),
) -> HistoryResponse:
    info = await run_in_threadpool(session_store.get_session_info, session.user_id)
    items = await run_in_threadpool(
        session_store.get_message_history, session.user_id, limit
    )

    return HistoryResponse(
        session_id=session.user_id,
        message_count=(info or {}).get("message_count", 0),
        first_seen_at=_iso((info or {}).get("first_seen_at")),
        last_seen_at=_iso((info or {}).get("last_seen_at")),
        items=[
            HistoryItem(
                question=item["question"],
                answer=item["answer"],
                created_at=_iso(item["created_at"]) or "",
            )
            for item in items
        ],
    )


@router.delete(
    "/history",
    summary="Joriy sessiya tarixini tozalash",
    description="Cookie orqali aniqlangan sessiyaning barcha xabarlarini o'chiradi.",
)
async def delete_history(session: Session = Depends(resolve_session)) -> dict:
    deleted = await run_in_threadpool(session_store.clear_history, session.user_id)
    return {
        "session_id": session.user_id,
        "deleted_messages": deleted,
        "detail": "Tarix tozalandi",
    }

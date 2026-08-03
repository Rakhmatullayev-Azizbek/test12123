"""
Umumiy FastAPI bog'liqliklari (dependencies).
"""
import uuid
from dataclasses import dataclass

from fastapi import Request, Response

from shared.config import settings


@dataclass
class Session:
    """Cookie orqali aniqlangan foydalanuvchi sessiyasi."""
    user_id: str
    is_new: bool


def resolve_session(request: Request, response: Response) -> Session:
    """
    Cookie'dan user_id ni o'qiydi. Yo'q bo'lsa (yangi qurilma/brauzer) yangi
    UUID yaratib, javobga cookie sifatida biriktiradi.

    Diqqat: cookie'ni darhol o'rnatamiz (endpoint muvaffaqiyatli tugashini
    kutmasdan), shunda xatolik bo'lgan so'rovdan keyin ham sessiya saqlanadi.

    Cross-origin frontend (masalan Vercel) uchun cookie ishlamasligi mumkin —
    shuning uchun avval `X-Session-Id` sarlavhasini tekshiramiz. U yaroqli UUID
    bo'lsa, sessiya shu bilan aniqlanadi (cookie'ga bog'liq emas).
    """
    header_id = request.headers.get("x-session-id")
    if header_id:
        try:
            return Session(user_id=str(uuid.UUID(header_id)), is_new=False)
        except ValueError:
            pass  # noto'g'ri format — cookie yo'liga tushamiz

    raw = request.cookies.get(settings.SESSION_COOKIE_NAME)

    # Cookie qiymati haqiqiy UUID bo'lishi kerak - aks holda soxta/buzilgan
    # qiymat bilan baza to'lib ketishi mumkin
    user_id: str | None = None
    if raw:
        try:
            user_id = str(uuid.UUID(raw))
        except ValueError:
            user_id = None

    is_new = user_id is None
    if is_new:
        user_id = str(uuid.uuid4())
        response.set_cookie(
            key=settings.SESSION_COOKIE_NAME,
            value=user_id,
            max_age=settings.SESSION_COOKIE_MAX_AGE_DAYS * 24 * 60 * 60,
            httponly=True,
            samesite=settings.SESSION_COOKIE_SAMESITE,
            secure=settings.SESSION_COOKIE_SECURE,
        )

    return Session(user_id=user_id, is_new=is_new)

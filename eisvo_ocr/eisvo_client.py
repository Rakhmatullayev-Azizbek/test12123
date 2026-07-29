"""EISVO API klienti — kiritilgan ID orqali shartnoma JSON'ini oladi (POST + Bearer).

JSON faylni yuklash o'rniga foydalanuvchi faqat ID kiritadi; server shu modul orqali
EISVO API'ga POST so'rov yuborib, javob JSON'ini oladi (u .json namunalar bilan bir xil
strukturada: {status, errorMsg, contract:{...}}).

Sozlamalar `.env` orqali (config.Settings, EISVO_ prefiksi):
  EISVO_API_URL         — POST endpoint to'liq URL
  EISVO_API_TOKEN       — Bearer token (muddati tugasa .env'da yangilab, web'ni restart)
  EISVO_API_ID_FIELD    — ID so'rov body'sida qaysi kalit bilan yuboriladi (default: idn)
  EISVO_API_BODY_FILE   — REQUIRED maydonlar JSON fayli yo'li (mas. /data/api_body.json)
  EISVO_API_BODY        — ...yoki inline JSON (fayl berilmasa)
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx

from .config import settings


class EisvoApiError(Exception):
    """API bilan bog'liq, foydalanuvchiga ko'rsatiladigan xato."""


def _body_template() -> str:
    """Body shablonining xom JSON matni (fayl yoki inline). Bo'sh bo'lsa ''."""
    if settings.eisvo_api_body_file.strip():
        p = Path(settings.eisvo_api_body_file.strip())
        if not p.is_file():
            raise EisvoApiError(f"EISVO_API_BODY_FILE topilmadi: {p}")
        return p.read_text(encoding="utf-8-sig")
    return settings.eisvo_api_body


def _build_body(contract_id: str) -> dict:
    """So'rov body'sini quradi. IKKI usul:
      (1) shablonda «{{id}}» placeholderi bo'lsa — ID o'sha joyga qo'yiladi (maydon nomi
          `contractorId` yoki boshqa bo'lishidan, istalgan chuqurlikda bo'lishidan qat'i
          nazar). Swagger'dagi butun body'ni nusxalab, ID o'rniga {{id}} yozing.
      (2) placeholder yo'q bo'lsa — ID yuqori darajaga `EISVO_API_ID_FIELD` kaliti bilan
          qo'shiladi (body shabloni faqat QOLGAN required maydonlar bo'ladi)."""
    raw = _body_template()
    if not raw.strip():
        return {settings.eisvo_api_id_field: contract_id}
    if "{{id}}" in raw:
        # placeholderni JSON-xavfsiz qo'yamiz (qo'shtirnoq ichida turadi)
        raw = raw.replace("{{id}}", json.dumps(str(contract_id))[1:-1])
        inject = False
    else:
        inject = True
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as e:
        raise EisvoApiError(f"REQUIRED body JSON yaroqsiz: {e}")
    if not isinstance(body, dict):
        raise EisvoApiError("REQUIRED body JSON obyekt (dict) bo'lishi kerak")
    if inject:
        body[settings.eisvo_api_id_field] = contract_id
    return body


def fetch_contract(contract_id: str) -> dict:
    """ID bo'yicha EISVO API'dan shartnoma JSON'ini oladi (POST + Bearer)."""
    if not settings.eisvo_api_url.strip():
        raise EisvoApiError("EISVO_API_URL sozlanmagan — .env fayliga qo'shing")

    body = _build_body(contract_id)

    headers = {"Accept": "application/json"}
    if settings.eisvo_api_token.strip():
        headers["Authorization"] = f"Bearer {settings.eisvo_api_token.strip()}"

    try:
        resp = httpx.post(
            settings.eisvo_api_url.strip(),
            json=body,
            headers=headers,
            timeout=settings.eisvo_api_timeout,
        )
    except httpx.RequestError as e:
        raise EisvoApiError(f"API'ga ulanib bo'lmadi: {e}")

    if resp.status_code in (401, 403):
        raise EisvoApiError(
            "Avtorizatsiya xatosi (HTTP %d) — Bearer token eskirgan yoki noto'g'ri "
            "(.env'dagi EISVO_API_TOKEN'ni yangilang)" % resp.status_code
        )
    if resp.status_code >= 400:
        raise EisvoApiError(f"API xatosi HTTP {resp.status_code}: {resp.text[:300]}")

    try:
        data = resp.json()
    except ValueError:
        raise EisvoApiError("API JSON qaytarmadi")
    if not isinstance(data, dict) or not (
        data.get("contract") or data.get("idn") or data.get("docNo")
    ):
        raise EisvoApiError(
            "API javobida shartnoma topilmadi — ID yoki required maydonlar noto'g'ri bo'lishi mumkin"
        )
    return data

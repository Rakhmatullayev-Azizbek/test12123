"""FastAPI backend — PDF + (ixtiyoriy) API JSON yuklab, natijani olish.

Ishga tushirish (docker):
  docker compose up -d web    ->  http://localhost:8080

Endpointlar:
  GET  /            — frontend (static/index.html)
  GET  /api/health  — surya/vllm serverlar holati
  POST /api/process — multipart: pdf (majburiy), api_json (ixtiyoriy)
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import eisvo_client
from .config import settings
from .pipeline import compare_with_api, process_pdf

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("eisvo_ocr.server")

app = FastAPI(title="EISVO LLM OCR", version="0.1.0")

# CORS — backend FAQAT quyidagi manbalarga javob beradi: Vercel'dagi frontend va
# lokal UI (localhost:8080). Boshqa sayt/origin bloklanadi. Ro'yxatni
# WEB_ALLOWED_ORIGINS (vergul bilan) orqali o'zgartirish mumkin.
_DEFAULT_ORIGINS = (
    "https://chat-zeta-sepia-25.vercel.app,"
    "http://localhost:8080,http://127.0.0.1:8080"
)
_ALLOWED_ORIGINS = [
    o.strip().rstrip("/")
    for o in os.environ.get("WEB_ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Origin darajasida majburlash: CORS faqat brauzerni cheklaydi (javob sarlavhasi),
# server esa baribir javob beradi. Bu middleware ruxsat etilmagan origin'li
# so'rovlarni (boshqa sayt/API) 403 bilan RAD etadi. Origin sarlavhasi YO'Q
# so'rovlar (sahifani to'g'ridan ochish, ichki health-check) o'tadi.
@app.middleware("http")
async def _restrict_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    if origin and origin.rstrip("/") not in _ALLOWED_ORIGINS:
        return JSONResponse(status_code=403, content={"detail": "Ruxsat etilmagan manba (origin)."})
    return await call_next(request)

_STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(_STATIC / "index.html")


@app.get("/api/health")
def health():
    def probe(url: str) -> bool:
        try:
            return httpx.get(url, timeout=3).status_code == 200
        except Exception:
            return False

    surya_url = os.environ.get("SURYA_INFERENCE_URL", "")
    surya_health = surya_url.removesuffix("/v1") + "/health" if surya_url else ""
    llm_health = settings.vllm_base_url.removesuffix("/v1") + "/health"
    return {
        "surya": probe(surya_health) if surya_health else None,
        "llm": probe(llm_health),
    }


@app.get("/api/config")
def api_config():
    """Frontend uchun: EISVO API ID-rejimi sozlanganmi (URL bor-yo'qligi)."""
    return {"api_lookup": bool(settings.eisvo_api_url.strip())}


# ── AI Chatbot (RAG) proksi ─────────────────────────────────────────────────
# Brauzer «AI Chatbot» bo'limidan /api/chat ga so'rov yuboradi (bir xil origin,
# 8080). Bu yerdan docker tarmog'idagi rag_backend'ga (8008) uzatiladi — shunda
# CORS/cookie muammosi bo'lmaydi va 8008 tashqariga ochiq bo'lishi shart emas.
CHAT_BACKEND_URL = os.environ.get("CHAT_BACKEND_URL", "http://rag_backend:8008").rstrip("/")
_CHAT_COOKIE = "tr_chatbot_uid"


@app.get("/api/chat/health")
async def api_chat_health():
    """Chatbot backend tayyorligi (web frontend indikatori uchun)."""
    try:
        async with httpx.AsyncClient(timeout=4) as client:
            r = await client.get(f"{CHAT_BACKEND_URL}/health")
        return {"reachable": r.status_code == 200, **(r.json() if r.headers.get(
            "content-type", "").startswith("application/json") else {})}
    except Exception:
        return {"reachable": False}


@app.get("/api/chat/stats")
async def api_chat_stats():
    """Admin panel uchun: indekslash statistikasi (rag_backend /stats proksi)."""
    try:
        async with httpx.AsyncClient(timeout=6) as client:
            r = await client.get(f"{CHAT_BACKEND_URL}/stats")
        return JSONResponse(status_code=r.status_code, content=r.json())
    except Exception as e:
        return JSONResponse(status_code=502, content={"detail": f"stats olinmadi: {e}"})


@app.post("/api/chat")
async def api_chat(request: Request):
    """Savolni rag_backend'ga uzatadi va sessiya cookie'sini ikki tomonlama olib o'tadi."""
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, "JSON body kutilyapti (masalan {\"question\": \"...\"})")

    fwd_cookies = {}
    if (c := request.cookies.get(_CHAT_COOKIE)):
        fwd_cookies[_CHAT_COOKIE] = c

    # Cross-origin (Vercel↔tunnel) cookie ishlamasligi mumkin — frontend barqaror
    # X-Session-Id yuboradi; uni rag_backend'ga o'tkazamiz (tarix shu bilan bog'lanadi).
    fwd_headers = {}
    if (sid := request.headers.get("x-session-id")):
        fwd_headers["X-Session-Id"] = sid

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(
                f"{CHAT_BACKEND_URL}/chat", json=payload,
                cookies=fwd_cookies, headers=fwd_headers,
            )
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Chatbot backendga ulanib bo'lmadi: {e}")

    ctype = r.headers.get("content-type", "")
    body = r.json() if ctype.startswith("application/json") else {"detail": r.text[:500]}
    resp = JSONResponse(status_code=r.status_code, content=body)

    # rag_backend bergan yangi sessiya cookie'sini brauzerga (8080 origin) o'tkazamiz
    if (new_cookie := r.cookies.get(_CHAT_COOKIE)):
        resp.set_cookie(
            _CHAT_COOKIE, new_cookie,
            max_age=365 * 24 * 3600, httponly=True, samesite="lax",
        )
    return resp


@app.post("/api/process")
def api_process(
    pdf: UploadFile = File(...),
    contract_id: str | None = Form(None),
    api_json: UploadFile | None = File(None),
):
    if not (pdf.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "PDF fayl kutilyapti")

    api_data = None
    # 1-ustuvor: kiritilgan ID bo'yicha EISVO API'dan olish
    if contract_id and contract_id.strip():
        try:
            api_data = eisvo_client.fetch_contract(contract_id.strip())
        except eisvo_client.EisvoApiError as e:
            raise HTTPException(502, f"EISVO API: {e}")
    # (eski/zaxira yo'l) to'g'ridan-to'g'ri JSON fayl yuklangan bo'lsa
    elif api_json is not None and api_json.filename:
        try:
            api_data = json.loads(api_json.file.read().decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise HTTPException(400, f"API JSON o'qib bo'lmadi: {e}")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf.file.read())
        tmp_path = tmp.name
    try:
        log.info("Yangi so'rov: %s (api_json=%s)", pdf.filename, bool(api_data))
        result = process_pdf(tmp_path)
    except Exception as e:  # OCR/LLM xatosi — frontendga tushunarli qaytaramiz
        log.exception("Pipeline xatosi")
        raise HTTPException(500, f"Pipeline xatosi: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    response = {
        "file": pdf.filename,
        "extracted": result["extracted"],
        "validation": result["validation"],
        "markdown": result["debug"]["markdown"],
    }
    if api_data is not None:
        cmp = compare_with_api(result, api_data)
        response["comparison"] = cmp["comparison"]
        response["confidence"] = cmp["confidence"]
    return response

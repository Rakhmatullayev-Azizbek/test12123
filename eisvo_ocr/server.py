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
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import eisvo_client
from .config import settings
from .pipeline import compare_with_api, process_pdf

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("eisvo_ocr.server")

app = FastAPI(title="EISVO LLM OCR", version="0.1.0")
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

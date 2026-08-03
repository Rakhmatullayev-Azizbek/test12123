"""
FastAPI ilovasi. Swagger UI: http://localhost:8008/docs
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.routers import chat, health, session
from app.services import session_store
from shared.config import settings
from shared.embedder import warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("app.main")

DESCRIPTION = """
O'zbekiston qonunchiligi bo'yicha savol-javob API (lex.uz hujjatlari asosida).

**Qanday ishlaydi:**

1. `scraper` — lex.uz'dan hujjatlarni yuklab `data/raw/` ga saqlaydi
2. `indexer` — hujjatlarni band/modda darajasida bo'laklab, `multilingual-e5-large`
   bilan vektorlashtiradi va Qdrant'ga yozadi
3. `rag_backend` (bu API) — savolni vektorlashtirib eng mos bandlarni topadi va
   ularni kontekst sifatida LLM'ga beradi

**Sessiya:** `tr_chatbot_uid` cookie avtomatik o'rnatiladi. Shu cookie orqali
suhbat tarixi saqlanadi va keyingi savollarda kontekst sifatida ishlatiladi
(kuzatuv savollari uchun).

**Boshlash:** avval `/stats` ni chaqirib indekslash tugaganini tekshiring
(`indexed_chunks > 0`), keyin `/chat` ni sinab ko'ring.
"""

TAGS_METADATA = [
    {"name": "chat", "description": "Savol-javob va vektor qidiruv."},
    {"name": "session", "description": "Sessiya va suhbat tarixi."},
    {"name": "health", "description": "Servislar holati va indekslash statistikasi."},
]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # --- ishga tushish ---
    try:
        session_store.init_db()
        logger.info("PostgreSQL jadvallari tayyor")
    except Exception:
        logger.exception("PostgreSQL'ni tayyorlashda xatolik - /chat ishlamasligi mumkin")

    # Embedding modelini oldindan yuklaymiz (2.2 GB). Aks holda birinchi
    # /chat so'rovi model yuklanishini kutib turadi va timeout bo'lishi mumkin.
    try:
        logger.info("Embedding modeli yuklanmoqda...")
        warmup()
        logger.info("Embedding modeli tayyor")
    except Exception:
        logger.exception("Embedding modelini yuklashda xatolik")

    yield

    # --- to'xtash ---
    session_store.close_pool()


app = FastAPI(
    title="TR Chatbot RAG API",
    description=DESCRIPTION,
    version="1.1.0",
    openapi_tags=TAGS_METADATA,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Boshqa loyihalar/frontendlar bu API'ga murojaat qilishi mumkin bo'lgani uchun CORS.
#
# DIQQAT: brauzerlar allow_origins=["*"] va allow_credentials=True juftligini
# rad etadi - bunda sessiya cookie'si hech qachon yuborilmaydi. Shuning uchun
# "*" berilganda credentials'ni o'chiramiz, aniq domenlar berilganda yoqamiz.
_allow_all = "*" in settings.ALLOWED_ORIGINS
if _allow_all:
    logger.warning(
        "ALLOWED_ORIGINS='*' - cookie'li sessiya cross-origin ishlamaydi. "
        "Production'da .env da aniq domenlarni ko'rsating."
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=not _allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(chat.router, tags=["chat"])
app.include_router(session.router, tags=["session"])


@app.get("/", include_in_schema=False)
def root():
    """Ildiz manzil - to'g'ridan-to'g'ri Swagger UI'ga yo'naltiradi."""
    return RedirectResponse(url="/docs")

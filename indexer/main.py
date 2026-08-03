"""
Indexer entrypoint.

Vazifasi: data/raw/*.html fayllarini o'qib, parslash, chunk'larga ajratish,
vektorlashtirish va Qdrant'ga yozish. Keyin metadata.db da hujjatni
'indexed' deb belgilash.

Ishlash tartibi:
    1. Qdrant kolleksiyasi mavjudligini kafolatlash
    2. metadata.db dan indekslanmagan hujjatlarni olish (indexed_at IS NULL)
    3. Har biri uchun: parse -> chunk -> embed -> Qdrant upsert -> mark_indexed
    4. INDEX_INTERVAL_MINUTES oralig'ida qaytadan tekshirish (scraper yangi
       hujjat qo'shgan bo'lishi mumkin)

Diqqat: ilgari bu fayl scraper/main.py ning nusxasi bo'lgan va indekslashni
umuman bajarmagan - shu sababli Qdrant kolleksiyasi hech qachon yaratilmagan.
"""
import logging
import signal
import sys
import time
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from shared.config import settings
from shared import metadata_db
from shared.embedder import embed_passages, warmup

from html_parser import parse_lex_document
import qdrant_writer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("indexer.main")

_shutdown = False


def index_document(document_id: str) -> int:
    """
    Bitta hujjatni indekslaydi va yozilgan chunk sonini qaytaradi.

    Xatolik bo'lsa exception ko'taradi - chaqiruvchi uni metadata.db ga yozadi.
    """
    html_path = Path(settings.RAW_DIR) / f"{document_id}.html"
    if not html_path.exists():
        raise FileNotFoundError(f"HTML fayl topilmadi: {html_path}")

    html = html_path.read_text(encoding="utf-8")
    parsed = parse_lex_document(html, document_id)
    chunks = parsed.to_chunks()

    if not chunks:
        logger.warning(f"{document_id}: chunk chiqmadi (hujjat bo'sh yoki format boshqa)")
        return 0

    # Eski chunk'larni tozalaymiz - hujjat yangilangan bo'lsa, olib tashlangan
    # bandlar Qdrant'da qolib ketmasligi kerak
    qdrant_writer.delete_document(document_id)

    total = 0
    batch_size = max(1, settings.EMBEDDING_BATCH_SIZE * 4)
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embed_passages([c["embed_text"] for c in batch])
        total += qdrant_writer.upsert_chunks(batch, vectors)

    bands = sum(1 for c in chunks if c["kind"] == "band")
    tables = sum(1 for c in chunks if c["kind"] == "jadval")
    logger.info(
        f"{document_id}: {total} chunk indekslandi "
        f"(band={bands}, jadval={tables}) — {parsed.document_title[:60]!r}"
    )
    return total


def run_index_cycle() -> None:
    logger.info("Indekslash sikli boshlandi")
    metadata_db.init_db(settings.METADATA_DB_PATH)
    qdrant_writer.ensure_collection()

    pending = metadata_db.list_pending(
        settings.METADATA_DB_PATH, include_all=settings.REINDEX_ALL
    )
    if not pending:
        logger.info("Indekslashni kutayotgan hujjat yo'q")
        return

    logger.info(f"{len(pending)} ta hujjat indekslanadi")

    ok = failed = total_chunks = 0
    for document_id in pending:
        if _shutdown:
            logger.info("To'xtatish signali - sikl uzildi")
            break
        try:
            count = index_document(document_id)
            metadata_db.mark_indexed(settings.METADATA_DB_PATH, document_id, count)
            total_chunks += count
            ok += 1
        except Exception as e:
            logger.exception(f"{document_id}: indekslashda xatolik")
            metadata_db.mark_failed(settings.METADATA_DB_PATH, document_id, f"{type(e).__name__}: {e}")
            failed += 1

    logger.info(
        f"Indekslash sikli tugadi. Muvaffaqiyatli={ok}, xato={failed}, "
        f"jami chunk={total_chunks}, Qdrant'dagi umumiy point={qdrant_writer.count_points()}"
    )


def _handle_signal(signum, _frame) -> None:
    global _shutdown
    logger.info(f"Signal {signum} qabul qilindi - to'xtatilmoqda")
    _shutdown = True


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # Modelni oldindan yuklaymiz: 2.2 GB, birinchi siklni kutib turmasligi uchun
    logger.info("Embedding modeli tayyorlanmoqda...")
    warmup()

    run_index_cycle()

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        run_index_cycle,
        "interval",
        minutes=settings.INDEX_INTERVAL_MINUTES,
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    logger.info(f"Scheduler ishga tushdi - har {settings.INDEX_INTERVAL_MINUTES} daqiqada")

    try:
        while not _shutdown:
            time.sleep(5)
    finally:
        scheduler.shutdown(wait=False)
        logger.info("Indexer to'xtadi")


if __name__ == "__main__":
    main()

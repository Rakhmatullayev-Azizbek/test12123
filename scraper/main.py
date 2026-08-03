"""
Scraper entrypoint.

document_urls.txt dagi hujjatlar ro'yxatini SCRAPE_INTERVAL_HOURS oralig'ida
qayta tekshiradi va yangi/o'zgargan hujjatlarni data/raw/ ga saqlaydi.
Indexer keyin metadata.db orqali indekslanmagan hujjatlarni topib oladi.

Diqqat: ilgari bu fayl list_fetcher.fetch_document_ids() ga tayanardi, lekin u
to'ldirilmagan stub bo'lib doim bo'sh ro'yxat qaytarardi - natijada scraper
har 12 soatda ishga tushib hech narsa qilmasdi. Endi manba - aniq URL ro'yxati.
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

from document_list import load_document_ids
from page_fetcher import fetch_document_html

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scraper.main")

# lex.uz'ni ortiqcha yuklamaslik uchun so'rovlar orasidagi pauza
REQUEST_DELAY_SECONDS = 1.0

_shutdown = False


def run_scrape_cycle() -> int:
    """
    Ro'yxatdagi hujjatlarni yuklab, yangi/o'zgarganlarini saqlaydi.
    Qaytaradi: saqlangan (indekslashga tayyor) hujjatlar soni.
    """
    logger.info("Scrape sikli boshlandi")
    metadata_db.init_db(settings.METADATA_DB_PATH)
    Path(settings.RAW_DIR).mkdir(parents=True, exist_ok=True)

    try:
        document_ids = load_document_ids(settings.DOCUMENT_LIST_PATH)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 0

    logger.info(f"Ro'yxatdan {len(document_ids)} ta hujjat ID o'qildi")

    saved = unchanged = failed = 0
    for doc_id in document_ids:
        if _shutdown:
            logger.info("To'xtatish signali - sikl uzildi")
            break

        html = fetch_document_html(settings.LEX_UZ_BASE_URL, doc_id)
        if html is None:
            failed += 1
            continue

        status = metadata_db.check_and_update(settings.METADATA_DB_PATH, doc_id, html)
        if status in ("new", "updated"):
            out_path = Path(settings.RAW_DIR) / f"{doc_id}.html"
            out_path.write_text(html, encoding="utf-8")
            saved += 1
            logger.info(f"{doc_id}: {status} - saqlandi")
        else:
            unchanged += 1

        time.sleep(REQUEST_DELAY_SECONDS)

    logger.info(
        f"Scrape sikli tugadi. Yangi/o'zgargan={saved}, o'zgarmagan={unchanged}, xato={failed}"
    )
    return saved


def _handle_signal(signum, _frame) -> None:
    global _shutdown
    logger.info(f"Signal {signum} qabul qilindi - to'xtatilmoqda")
    _shutdown = True


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # Ishga tushganda darhol bir marta ishlaydi, keyin jadval bo'yicha davom etadi
    run_scrape_cycle()

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        run_scrape_cycle,
        "interval",
        hours=settings.SCRAPE_INTERVAL_HOURS,
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    logger.info(f"Scheduler ishga tushdi - har {settings.SCRAPE_INTERVAL_HOURS} soatda")

    try:
        while not _shutdown:
            time.sleep(5)
    finally:
        scheduler.shutdown(wait=False)
        logger.info("Scraper to'xtadi")


if __name__ == "__main__":
    main()

"""
Hujjatlar ro'yxatini bir marta (scheduler'siz) yuklab olish uchun CLI vositasi.

Ishlatilishi:
    python scrape_from_list.py                      # DOCUMENT_LIST_PATH dan
    python scrape_from_list.py document_urls.txt    # aniq fayldan

Asosiy scrape mantiqi main.py da - bu fayl faqat qo'lda bir marta ishga
tushirish uchun qulay o'ram.
"""
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from shared.config import settings

import main as scraper_main

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scraper.scrape_from_list")


def main() -> None:
    if len(sys.argv) > 2:
        print(__doc__)
        sys.exit(1)

    if len(sys.argv) == 2:
        # Ro'yxat yo'lini vaqtincha almashtiramiz
        settings.DOCUMENT_LIST_PATH = sys.argv[1]

    logger.info(f"Ro'yxat: {settings.DOCUMENT_LIST_PATH}")
    saved = scraper_main.run_scrape_cycle()
    logger.info(f"Tugadi. Indekslashga tayyor: {saved} ta hujjat")


if __name__ == "__main__":
    main()

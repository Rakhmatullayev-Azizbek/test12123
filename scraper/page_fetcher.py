"""
Bitta lex.uz hujjat sahifasini (/docs/{id}) yuklab olish.
"""
import logging
import time

import httpx

logger = logging.getLogger("scraper.page_fetcher")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TR-Chatbot-Bot/1.0; +internal use)",
    "Accept-Language": "uz,ru;q=0.8,en;q=0.5",
}

MAX_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0

# Hujjat sahifasi shu belgini o'z ichiga olishi kerak - aks holda
# lex.uz xato/bo'sh sahifa qaytargan bo'ladi
_CONTENT_MARKER = 'id="divCont"'


def fetch_document_html(
    base_url: str,
    document_id: str,
    timeout: float = 30.0,
) -> str | None:
    """
    document_id: masalan "-2818868" (salbiy raqam, lex.uz'ning o'ziga xos ID formati)
    Qaytaradi: HTML matn, yoki xatolik bo'lsa None.

    Vaqtinchalik tarmoq xatolarida MAX_ATTEMPTS marta qayta uriniladi.
    """
    url = f"{base_url}/docs/{document_id}"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with httpx.Client(
                headers=HEADERS, timeout=timeout, follow_redirects=True
            ) as client:
                response = client.get(url)
                response.raise_for_status()
                html = response.text

            if _CONTENT_MARKER not in html:
                logger.warning(
                    f"{document_id}: sahifada divCont yo'q "
                    f"({len(html)} bayt) - hujjat mavjud emas yoki format o'zgargan"
                )
                return None

            return html

        except httpx.HTTPError as e:
            if attempt < MAX_ATTEMPTS:
                wait = RETRY_BACKOFF_SECONDS * attempt
                logger.warning(
                    f"{document_id}: {attempt}/{MAX_ATTEMPTS} urinish muvaffaqiyatsiz "
                    f"({e}) - {wait:.0f}s dan keyin qayta urinaman"
                )
                time.sleep(wait)
            else:
                logger.error(f"{document_id}: yuklab olishda xatolik - {e}")

    return None

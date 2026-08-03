"""
Hujjatlar ro'yxatini (URL yoki ID) fayldan o'qish.

Qo'llab-quvvatlanadigan formatlar (har qatorda bittasi):
    https://lex.uz/uz/docs/-2818868
    https://lex.uz/docs/-4975566?ONDATE=01.04.2025
    https://lex.uz/acts/-2818868
    -2818868
    # izoh qatori (o'tkazib yuboriladi)
"""
import logging
import re
from pathlib import Path

logger = logging.getLogger("scraper.document_list")

# lex.uz hujjat havolasidan ID: /docs/-2818868 yoki /acts/-2818868
_URL_ID_PATTERN = re.compile(r"/(?:docs|acts)/(-?\d+)")
_BARE_ID_PATTERN = re.compile(r"^-?\d+$")


def extract_document_id(line: str) -> str | None:
    """URL yoki ID qatoridan document_id'ni ajratib oladi."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    match = _URL_ID_PATTERN.search(line)
    if match:
        return match.group(1)

    if _BARE_ID_PATTERN.match(line):
        return line

    logger.warning(f"Tanib bo'lmadi, o'tkazib yuborildi: {line!r}")
    return None


def extract_ids_from_html(html: str) -> list[str]:
    """Sahifa HTML'idan /docs/{id} havolalarini ajratib oladi (kelajakda avtomatik yig'ish uchun)."""
    return sorted(set(_URL_ID_PATTERN.findall(html)))


def load_document_ids(list_path: str) -> list[str]:
    """
    Fayldan hujjat ID'larini o'qiydi. Takrorlar olib tashlanadi, tartib saqlanadi.
    """
    path = Path(list_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Hujjatlar ro'yxati topilmadi: {list_path}. "
            f"DOCUMENT_LIST_PATH sozlamasini tekshiring."
        )

    ids: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        doc_id = extract_document_id(line)
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            ids.append(doc_id)
    return ids

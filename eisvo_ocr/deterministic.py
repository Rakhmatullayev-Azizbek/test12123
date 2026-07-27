"""[4] DETERMINISTIK EXTRACTION (LLM'gacha).

  - regex: INN (9 raqam), MFO (5), hisob raqam (20), SWIFT, sana, shartnoma raqami
  - normalizatsiya: summalar, sanalar, Incoterms (DAT->DPU)

Bu qatlam LLM'ga «hint» beradi va validatorda format tekshiruvi uchun ishlatiladi.
"""
from __future__ import annotations

import re
from datetime import date

# --- format regexlari (validator ham shulardan foydalanadi) ---
RE_INN = re.compile(r"(?<!\d)\d{9}(?!\d)")
RE_MFO = re.compile(r"(?<!\d)\d{5}(?!\d)")
RE_ACCOUNT = re.compile(r"(?<!\d)\d{20}(?!\d)")
RE_SWIFT = re.compile(r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b")
RE_DATE = re.compile(
    r"\b(\d{1,2})[./\-](\d{1,2})[./\-](\d{4})\b"          # 25.03.2024
    r"|\b(\d{4})[./\-](\d{1,2})[./\-](\d{1,2})\b"          # 2024-03-25
)
RE_CONTRACT_NO = re.compile(
    r"(?:договор|контракт|contract|shartnoma|kontrakt|№)[^\S\n]*(?:№|No\.?|N|#)?[^\S\n]*"
    r"([A-Za-zА-Яа-я0-9][A-Za-zА-Яа-я0-9/\-\.]{0,30})",
    re.IGNORECASE,
)
RE_AMOUNT = re.compile(r"(?<![\d.,])\d{1,3}(?:[  ]?\d{3})*(?:[.,]\d{1,2})?(?![\d])")

# eski Incoterms -> amaldagi (Incoterms 2020)
INCOTERMS_MAP = {"DAT": "DPU", "DAF": "DAP", "DES": "DAP", "DEQ": "DPU", "DDU": "DAP"}
INCOTERMS_ALL = {
    "EXW", "FCA", "FAS", "FOB", "CFR", "CIF", "CPT", "CIP", "DAP", "DPU", "DDP",
} | set(INCOTERMS_MAP)
RE_INCOTERMS = re.compile(r"\b(" + "|".join(INCOTERMS_ALL) + r")\b")

_MONTHS = {
    # ru
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
    # uz
    "yanvar": 1, "fevral": 2, "mart": 3, "aprel": 4, "may": 5, "iyun": 6,
    "iyul": 7, "avgust": 8, "sentyabr": 9, "oktyabr": 10, "noyabr": 11, "dekabr": 12,
    # en
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
RE_DATE_WORDS = re.compile(
    r"\b(\d{1,2})\s*[-«»\"']*\s*(" + "|".join(_MONTHS) + r")\w*\s+(\d{4})", re.IGNORECASE
)


def normalize_amount(raw: str) -> float | None:
    """'1 234 567,89' / '1,234,567.89' / '1234.5' -> float."""
    s = raw.strip().replace(" ", " ").replace(" ", "")
    if not s:
        return None
    # oxirgi ajratgich — kasr; qolganlari minglik
    last_dot, last_com = s.rfind("."), s.rfind(",")
    dec = max(last_dot, last_com)
    if dec != -1 and len(s) - dec - 1 in (1, 2):
        int_part = re.sub(r"[.,]", "", s[:dec])
        s = int_part + "." + s[dec + 1:]
    else:
        s = re.sub(r"[.,]", "", s)
    try:
        return float(s)
    except ValueError:
        return None


def normalize_date(raw: str) -> str | None:
    """Har xil formatdagi sanani ISO (YYYY-MM-DD) ga keltirish."""
    m = RE_DATE.search(raw)
    if m:
        if m.group(1):
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        else:
            y, mo, d = int(m.group(4)), int(m.group(5)), int(m.group(6))
        try:
            return date(y, mo, d).isoformat()
        except ValueError:
            return None
    m = RE_DATE_WORDS.search(raw.lower())
    if m:
        mo = _MONTHS.get(m.group(2)[:8]) or _MONTHS.get(m.group(2))
        if mo:
            try:
                return date(int(m.group(3)), mo, int(m.group(1))).isoformat()
            except ValueError:
                return None
    return None


def normalize_incoterms(raw: str) -> str | None:
    m = RE_INCOTERMS.search(raw.upper())
    if not m:
        return None
    term = m.group(1)
    return INCOTERMS_MAP.get(term, term)


def _by_context(text: str, regex: re.Pattern, keywords: tuple[str, ...]) -> list[str]:
    """Regex nomzodlarini kalit so'zli qatorlardan afzal ko'rib yig'ish.
    Kalit so'zli qatorda topilsa — faqat shular (soxta mosliklar chiqib ketadi:
    masalan summalar MFO'ga, OCR buzgan sarlavha SWIFT'ga o'xshaydi);
    hech biri topilmasa — barcha format-mos nomzodlar (label OCR'da buzilgan holat)."""
    ctx: list[str] = []
    everything: list[str] = []
    for line in text.splitlines():
        found = regex.findall(line)
        if not found:
            continue
        everything.extend(found)
        if any(k in line.lower() for k in keywords):
            ctx.extend(found)
    return list(dict.fromkeys(ctx if ctx else everything))


def extract_candidates(text: str) -> dict:
    """Butun matndan deterministik nomzodlarni yig'ish — LLM promptiga hint sifatida
    va validatorda o'zaro tekshiruv uchun."""
    accounts = list(dict.fromkeys(RE_ACCOUNT.findall(text)))
    # 20 xonali hisob raqamlar ichidagi 9/5 xonali bo'laklarni INN/MFO deb olmaslik
    # uchun avval hisoblarni olib tashlaymiz
    text_wo_accounts = RE_ACCOUNT.sub(" ", text)
    dates = []
    for m in RE_DATE.finditer(text):
        iso = normalize_date(m.group(0))
        if iso:
            dates.append(iso)
    for m in RE_DATE_WORDS.finditer(text.lower()):
        iso = normalize_date(m.group(0))
        if iso:
            dates.append(iso)

    contract_numbers = [
        m.group(1).rstrip(".").strip()
        for m in RE_CONTRACT_NO.finditer(text)
        if m.group(1)
        and not m.group(1).lower().startswith(("от", "dan"))
        and any(ch.isdigit() for ch in m.group(1))  # "контракта:" dagi "а" kabi qoldiqlar emas
    ]

    return {
        "inn": _by_context(text_wo_accounts, RE_INN, ("инн", "inn", "стир", "stir")),
        "mfo": _by_context(text_wo_accounts, RE_MFO, ("мфо", "mfo")),
        "accounts": accounts,
        "swift": _by_context(text, RE_SWIFT, ("swift", "свифт", "bic", "бик")),
        "dates": list(dict.fromkeys(dates)),
        "contract_numbers": list(dict.fromkeys(contract_numbers)),
        "incoterms": list(dict.fromkeys(
            normalize_incoterms(m.group(0)) for m in RE_INCOTERMS.finditer(text.upper())
        )),
    }

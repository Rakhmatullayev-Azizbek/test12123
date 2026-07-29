"""[6] VALIDATOR QATLAMI (veto + confidence).

  - arifmetika: sum(qty*price) ~= amount, sum(amount) ~= total
  - format: INN/MFO/hisob/SWIFT regex'ga mosligi
  - OCR-match: LLM chiqargan qiymat OCR matnida bormi (hallucination veto)
  - har maydonga confidence: high / low / failed
"""
from __future__ import annotations

import math
import re
from typing import Any

from rapidfuzz import fuzz

from .config import settings
from .deterministic import RE_ACCOUNT, RE_INN, RE_MFO, RE_SWIFT
from .schemas import ExtractedContract

HIGH, LOW, FAILED, NA = "high", "low", "failed", "na"

_FORMAT_CHECKS = {
    "inn": lambda v: bool(RE_INN.fullmatch(v)),
    "mfo": lambda v: bool(RE_MFO.fullmatch(v)),
    "account": lambda v: bool(RE_ACCOUNT.fullmatch(v)),
    "swift": lambda v: bool(RE_SWIFT.fullmatch(v)),
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _in_ocr(value: Any, ocr_text_norm: str, ocr_digits: str) -> bool:
    """LLM qiymati OCR matnida uchraydimi (fuzzy). Sonlar — raqam ketma-ketligi bo'yicha."""
    if value is None:
        return True
    s = str(value)
    if isinstance(value, float):
        # 1500.0 -> "1500" / "1500.00" ko'rinishlarini ham qidiramiz
        digits = re.sub(r"\D", "", f"{value:.2f}".rstrip("0").rstrip("."))
        return bool(digits) and digits in ocr_digits
    sn = _norm(s)
    if not sn:
        return True
    if sn in ocr_text_norm:
        return True
    if sn.isdigit():
        return sn in ocr_digits
    return fuzz.partial_ratio(sn, ocr_text_norm) >= settings.ocr_match_min_ratio * 100


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=settings.arithmetic_rel_tolerance, abs_tol=0.05)


def validate(extracted: ExtractedContract, ocr_plain_text: str) -> dict:
    """Har maydon uchun {"value", "confidence", "issues"} qaytaradi."""
    ocr_norm = _norm(ocr_plain_text)
    ocr_digits = re.sub(r"\D", "", ocr_plain_text)
    fields: dict[str, dict] = {}

    def add(path: str, value: Any, fmt_key: str | None = None, skip_ocr: bool = False,
            required: bool = True):
        """Topilmagan (None) maydon — neytral «topilmadi» (NA), qizil xato EMAS.
        `required` faqat umumiy ishonch baliga ta'sir qiladi: majburiy maydon
        topilmasa (shartnoma raqami/summa/tomon nomi) verdict pasayadi, ixtiyoriy
        maydon (chet el INN, MFO, bank...) topilmasa neytral qoladi. Har ikkisi ham
        «Topilmadi» ro'yxatida ko'rinadi."""
        issues: list[str] = []
        conf = HIGH
        if value is None:
            fields[path] = {"value": None, "confidence": NA,
                            "issues": ["topilmadi"], "required": required}
            return
        # format tekshiruvi raqamlar orasidagi bo'shliqlarni e'tiborga olmaydi
        # (INN «211 212 877», hisob «8787 8787 ...» — bular to'g'ri, faqat guruhlangan)
        if fmt_key and not _FORMAT_CHECKS[fmt_key](re.sub(r"\s+", "", str(value))):
            issues.append(f"format xato ({fmt_key})")
            conf = FAILED
        if not skip_ocr and not _in_ocr(value, ocr_norm, ocr_digits):
            issues.append("OCR matnida topilmadi (hallucination xavfi)")
            conf = FAILED if conf == FAILED else LOW
        fields[path] = {"value": value, "confidence": conf, "issues": issues}

    # majburiy maydonlar (topilmasa haqiqatan muammo): shartnoma raqami/sanasi,
    # valyuta, summa, har ikki tomon nomi. Qolganlari ixtiyoriy (required=False):
    # bo'lmasa jim, MAVJUD bo'lsa format/OCR jihatidan tekshiriladi.
    m = extracted.meta
    add("contract_number", m.contract_number)
    add("contract_date", m.contract_date, skip_ocr=True)  # ISO'ga normalizatsiya qilingan
    add("currency", m.currency)
    add("total_amount", m.total_amount)
    add("payment_terms", m.payment_terms, skip_ocr=True, required=False)
    for role in ("seller", "buyer"):
        party = getattr(m, role)
        if party is None:
            fields[role] = {"value": None, "confidence": NA,
                            "issues": ["topilmadi"], "required": True}
            continue
        add(f"{role}.name", party.name)
        add(f"{role}.country", party.country, required=False)
        add(f"{role}.address", party.address, required=False)
        add(f"{role}.inn", party.inn, fmt_key="inn", required=False)
        add(f"{role}.director", party.director, required=False)
        if party.bank:
            add(f"{role}.bank.name", party.bank.name, required=False)
            add(f"{role}.bank.account", party.bank.account, fmt_key="account", required=False)
            add(f"{role}.bank.mfo", party.bank.mfo, fmt_key="mfo", required=False)
            add(f"{role}.bank.swift", party.bank.swift, fmt_key="swift", required=False)
    if getattr(m, "receiver", None):
        add("receiver.name", m.receiver.name, required=False)
        add("receiver.country", m.receiver.country, required=False)
        add("receiver.address", m.receiver.address, required=False)
        add("receiver.inn", m.receiver.inn, required=False)  # chet el qabul qiluvchi 9-raqamli emas
    if m.delivery:
        add("delivery.incoterms", m.delivery.incoterms, skip_ocr=True, required=False)
        add("delivery.place", m.delivery.place, required=False)

    # --- arifmetika ---
    products = extracted.products.products
    row_issues = []
    line_sum = 0.0
    all_amounts_ok = True
    for i, p in enumerate(products):
        add(f"products[{i}].name", p.name)
        add(f"products[{i}].quantity", p.quantity)
        add(f"products[{i}].price", p.price)
        add(f"products[{i}].amount", p.amount)
        if p.quantity is not None and p.price is not None and p.amount is not None:
            if not _close(p.quantity * p.price, p.amount):
                row_issues.append(
                    f"qator {i}: {p.quantity}x{p.price}={p.quantity * p.price:.2f} != {p.amount}"
                )
                fields[f"products[{i}].amount"]["confidence"] = LOW
                fields[f"products[{i}].amount"]["issues"].append("qty*price mos emas")
                all_amounts_ok = False
        if p.amount is not None:
            line_sum += p.amount

    totals_to_check = [
        t for t in (extracted.products.total_amount, m.total_amount) if t is not None
    ]
    total_ok = None
    if products and totals_to_check:
        total_ok = any(_close(line_sum, t) for t in totals_to_check)
        if not total_ok:
            for path in ("total_amount",):
                if path in fields and fields[path]["value"] is not None:
                    fields[path]["confidence"] = LOW
                    fields[path]["issues"].append(
                        f"sum(amount)={line_sum:.2f} total bilan mos emas"
                    )

    counts = {HIGH: 0, LOW: 0, FAILED: 0, NA: 0}
    crit_missing = 0  # topilmagan MAJBURIY maydonlar (verdict'ga ta'sir qiladi)
    for f in fields.values():
        counts[f["confidence"]] = counts.get(f["confidence"], 0) + 1
        if f["confidence"] == NA and f.get("required"):
            crit_missing += 1
    # neytral (ixtiyoriy topilmadi) maydonlar ratioga kirmaydi
    n = max(counts[HIGH] + counts[LOW] + counts[FAILED] + crit_missing, 1)
    failed_like = counts[FAILED] + crit_missing
    overall = HIGH if failed_like == 0 and counts[LOW] / n <= 0.1 else (
        LOW if failed_like / n <= 0.2 else FAILED
    )

    return {
        "fields": fields,
        "arithmetic": {
            "rows_ok": all_amounts_ok,
            "row_issues": row_issues,
            "line_sum": round(line_sum, 2),
            "total_ok": total_ok,
        },
        "summary": {**counts, "overall": overall, "critical_missing": crit_missing},
    }

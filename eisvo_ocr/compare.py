"""[7] compare_contract_data() — PDF'dan ajratilgan ma'lumot vs EISVO API JSON'i.

API strukturasi (real namunalar asosida): javob ildizida {"contract": {...}},
tomonlar contractorUz*/contractorFor*, tovarlar specifications[].goods[],
kodlar klassifikatorda (currCode "840", incotermsCode "08" ...) — tarjima codes.py da.

Import shartnomada Uz tomon = buyer; eksportda aksincha. Yo'nalish avtomatik
aniqlanadi (ikkala variant sinab, mosligi ko'prog'i olinadi).
"""
from __future__ import annotations

import re
from typing import Any, Callable

from rapidfuzz import fuzz

from .codes import (
    _translit,
    country_matches,
    currency_matches,
    incoterms_matches,
    unit_matches,
)
from .config import settings
from .deterministic import normalize_amount, normalize_date

# (bizning yo'l, API yo'li "contract" ichida, taqqoslash turi)
# {P} — tomon prefiksi: import'da seller=For/buyer=Uz, eksportda aksincha
_FIELD_SPECS: list[tuple[str, str, str]] = [
    ("contract_number", "docNo", "id"),
    ("contract_date", "docDate", "date"),
    ("currency", "currCode1", "currency"),
    ("total_amount", "amount1", "number"),
    ("{FOR}.name", "contractorForName", "text"),
    ("{FOR}.country", "contractorForCountryCode", "country"),
    ("{FOR}.address", "contractorForAddress", "text"),
    ("{FOR}.bank.name", "bankForName", "text"),
    # SWIFT + hisob API'da yagona erkin matn (bankForAttributes) — birlashtirib
    # tekshiriladi (pastdagi _bank_attributes_row), shuning uchun bu yerda alohida emas
    ("{UZ}.name", "contractorUzName", "text"),
    ("{UZ}.inn", "contractorUzInn", "id"),
    ("{UZ}.address", "contractorUzAddress", "text"),
    ("delivery.place", "deliveryTerms.0.destination", "text"),
    ("delivery.incoterms", "deliveryTerms.0.incotermsCode", "incoterms"),
]

# yo'nalishni aniqlashda ishlatiladigan maydonlar
_ORIENTATION_FIELDS = {"{FOR}.name", "{UZ}.name", "{UZ}.inn", "{FOR}.country"}


def _get_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(cur, dict):
            cur = cur.get(part)
        else:
            cur = getattr(cur, part, None)
    return cur


# tashkiliy-huquqiy shakllar (ru/uz/en) — nom solishtirishda tashlab yuboriladi:
# LLC == MCHJ == OOO bir xil ma'no, taqqoslashga xalaqit bermasin
# (transliteratsiya kirill/lotin МЕГАТОРГ==MEGATORG uchun codes._translit'da)
_LEGAL_FORMS = {
    "ooo", "oao", "zao", "ao", "pao", "chp", "ip", "npo", "aj", "xk",
    "mchj", "yatt", "qmj", "jsc",
    "llc", "ltd", "co", "inc", "corp", "gmbh", "ag", "sa", "srl", "spa",
    "bv", "plc", "pte", "pvt",
}


def _norm_text(s: Any) -> str:
    s = _translit(str(s).lower())
    # tinish belgilarini (defis, qo'shtirnoq, nuqta, apostrof...) bo'sh joyga
    s = re.sub(r"[^0-9a-z\s]", " ", s)
    return " ".join(t for t in s.split() if t not in _LEGAL_FORMS)


def _num(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    if v is None:
        return None
    return normalize_amount(str(v))


def _cmp_number(ours: Any, theirs: Any) -> tuple[str, float]:
    a, b = _num(ours), _num(theirs)
    if a is None or b is None:
        return "mismatch", 0.0
    ok = abs(a - b) <= max(abs(b) * 0.005, 0.05)  # API yaxlitlashi uchun 0.5%
    return ("match" if ok else "mismatch"), (1.0 if ok else 0.0)


def _cmp_date(ours: Any, theirs: Any) -> tuple[str, float]:
    a = normalize_date(str(ours)) or str(ours).strip()
    b = normalize_date(str(theirs)) or str(theirs).strip()
    return ("match" if a == b else "mismatch"), (1.0 if a == b else 0.0)


def _cmp_id(ours: Any, theirs: Any) -> tuple[str, float]:
    a = re.sub(r"[\s\-–—]", "", str(ours)).upper().lstrip("№#")
    b = re.sub(r"[\s\-–—]", "", str(theirs)).upper().lstrip("№#")
    return ("match" if a == b else "mismatch"), (1.0 if a == b else 0.0)


def _cmp_text(ours: Any, theirs: Any) -> tuple[str, float]:
    score = fuzz.token_sort_ratio(_norm_text(ours), _norm_text(theirs)) / 100
    ok = score >= settings.compare_fuzzy_threshold
    return ("match" if ok else "mismatch"), round(score, 3)


def _cmp_contains(ours: Any, theirs: Any) -> tuple[str, float]:
    """PDF qiymati API'ning erkin matn maydonida bormi (SWIFT/hisob bankForAttributes ichida)."""
    a = re.sub(r"\s", "", str(ours)).upper()
    b = re.sub(r"\s", "", str(theirs)).upper()
    if not a:
        return "mismatch", 0.0
    ok = a in b
    return ("match" if ok else "mismatch"), (1.0 if ok else 0.0)


def _code_cmp(fn: Callable) -> Callable:
    def inner(ours: Any, theirs: Any) -> tuple[str, float]:
        res = fn(str(ours), str(theirs))
        if res is None:
            return "unknown_code", 0.0
        return ("match" if res else "mismatch"), (1.0 if res else 0.0)

    return inner


_COMPARATORS: dict[str, Callable] = {
    "text": _cmp_text,
    "number": _cmp_number,
    "date": _cmp_date,
    "id": _cmp_id,
    "contains": _cmp_contains,
    "currency": _code_cmp(currency_matches),
    "country": _code_cmp(country_matches),
    "incoterms": _code_cmp(incoterms_matches),
}


def _bank_attributes_row(extracted: dict, contract: dict, for_side: str) -> dict:
    """FOR (chet el) tomon bank rekvizitlari — API'da hammasi bitta erkin matn
    (bankForAttributes: 'SWIFT CODE: ... No: ...'). PDF'dagi SWIFT va hisobni
    birga ko'rsatib, ikkalasi ham shu matn ichida borligini tekshiramiz."""
    bank = _get_path(extracted, f"{for_side}.bank")
    swift = account = ""
    if isinstance(bank, dict):
        swift = str(bank.get("swift") or "").strip()
        account = str(bank.get("account") or "").strip()
    api_attrs = contract.get("bankForAttributes")
    needed = [v for v in (swift, account) if v]
    pdf_display = "; ".join(
        p for p in (f"SWIFT: {swift}" if swift else "",
                    f"hisob: {account}" if account else "") if p
    )
    if not needed:
        status, score = "missing_pdf", 0.0
    elif not (api_attrs and str(api_attrs).strip()):
        status, score = "missing_api", 0.0
    else:
        hay = re.sub(r"\s", "", str(api_attrs)).upper()
        present = sum(1 for v in needed if re.sub(r"\s", "", v).upper() in hay)
        status = "match" if present == len(needed) else "mismatch"
        score = round(present / len(needed), 3)
    return {
        "pdf": pdf_display or None, "api": api_attrs,
        "api_field": "bankForAttributes", "status": status, "score": score,
        "_tpl": f"{for_side}.bank.attributes",  # yo'nalish skoriga kirmaydi
    }


def _compare_fields(extracted: dict, contract: dict, orientation: str) -> dict[str, dict]:
    for_side, uz_side = ("seller", "buyer") if orientation == "import" else ("buyer", "seller")
    out: dict[str, dict] = {}
    for our_tpl, api_path, kind in _FIELD_SPECS:
        our_path = our_tpl.replace("{FOR}", for_side).replace("{UZ}", uz_side)
        ours = _get_path(extracted, our_path)
        theirs = _get_path(contract, api_path)
        if ours is None or (isinstance(ours, str) and not ours.strip()):
            status, score = "missing_pdf", 0.0
        elif theirs is None or (isinstance(theirs, str) and not theirs.strip()):
            status, score = "missing_api", 0.0
        else:
            status, score = _COMPARATORS[kind](ours, theirs)
        out[our_path] = {
            "pdf": ours, "api": theirs, "api_field": api_path,
            "status": status, "score": score, "_tpl": our_tpl,
        }
    out[f"{for_side}.bank.attributes"] = _bank_attributes_row(extracted, contract, for_side)
    return out


def _detect_orientation(extracted: dict, contract: dict) -> tuple[str, dict[str, dict]]:
    """Import/eksportni aniqlash: qaysi biriktirishda nomlar ko'proq mos kelsa — o'sha."""
    results = {o: _compare_fields(extracted, contract, o) for o in ("import", "export")}
    scores = {
        o: sum(f["score"] for f in fields.values() if f["_tpl"] in _ORIENTATION_FIELDS)
        for o, fields in results.items()
    }
    orientation = max(scores, key=scores.get)  # teng bo'lsa import (tez-tez uchraydigani)
    fields = results[orientation]
    for f in fields.values():
        f.pop("_tpl", None)
    return orientation, fields


# ---------------- Tovarlar ----------------

def _split_bilingual(name: str) -> list[str]:
    """'bolts / Болт' -> ['bolts / Болт', 'bolts', 'Болт']."""
    parts = [p.strip() for p in re.split(r"\s*/\s*", name) if p.strip()]
    return [name] + parts if len(parts) > 1 else [name]


def _name_score(pdf_name: str, api_name: str) -> float:
    """token_set_ratio ham hisoblanadi: API tovar nomini qisqartirib beradi
    ('Баклажан свежий' PDF, 'Баклажан' API) — bunda bir nom ikkinchisining
    kichik to'plami, token_sort past baho beradi, token_set esa to'g'ri."""
    a = _norm_text(pdf_name)
    best = 0.0
    for v in _split_bilingual(api_name):
        b = _norm_text(v)
        best = max(best, fuzz.token_sort_ratio(a, b), fuzz.token_set_ratio(a, b))
    return best / 100


def _compare_products(products: list[dict], contract: dict) -> dict:
    goods = [
        g
        for spec in contract.get("specifications") or []
        for g in spec.get("goods") or []
    ]
    matched, used = [], set()
    for p in products:
        best_j, best_score = None, 0.0
        for j, g in enumerate(goods):
            if j in used:
                continue
            score = _name_score(p.get("name") or "", g.get("itemsName") or "")
            if p.get("hs_code") and str(p["hs_code"]) == str(g.get("tnCode")):
                score += 0.3
            qa, qb = _num(p.get("quantity")), _num(g.get("quantity"))
            if qa is not None and qb is not None and abs(qa - qb) <= abs(qb) * 0.001:
                score += 0.2
            if score > best_score:
                best_j, best_score = j, score
        if best_j is not None and best_score >= 0.6:
            used.add(best_j)
            g = goods[best_j]
            checks = {
                "name": _cmp_text(p.get("name"), g.get("itemsName"))
                if p.get("name") and g.get("itemsName") else ("missing_pdf", 0.0),
                "hs_code": _cmp_id(p.get("hs_code"), g.get("tnCode"))
                if p.get("hs_code") else ("missing_pdf", 0.0),
                "quantity": _cmp_number(p.get("quantity"), g.get("quantity")),
                "price": _cmp_number(p.get("price"), g.get("cost")),
                "amount": _cmp_number(p.get("amount"), g.get("amount")),
            }
            if p.get("unit"):
                res = unit_matches(p["unit"], str(g.get("unitCode")))
                checks["unit"] = (
                    ("unknown_code", 0.0) if res is None
                    else (("match" if res else "mismatch"), 1.0 if res else 0.0)
                )
            # nom fuzzy chegarasi tovarlar uchun yumshoqroq (ikki tilli yozuvlar)
            nm_status, nm_score = checks["name"]
            if nm_status == "mismatch" and _name_score(
                p.get("name") or "", g.get("itemsName") or ""
            ) >= 0.7:
                checks["name"] = ("match", nm_score)
            matched.append({
                "pdf": p,
                "api": {k: g.get(k) for k in
                        ("serialNo", "tnCode", "itemsName", "unitCode", "quantity", "cost", "amount")},
                "fields": {k: {"status": s, "score": sc} for k, (s, sc) in checks.items()},
            })
        else:
            matched.append({"pdf": p, "api": None, "fields": {}, "status": "unmatched_pdf"})

    unmatched_api = [
        {k: g.get(k) for k in ("serialNo", "tnCode", "itemsName", "quantity", "cost", "amount")}
        for j, g in enumerate(goods)
        if j not in used
    ]

    pdf_total = sum(a for p in products if (a := _num(p.get("amount"))) is not None)
    api_total = sum(a for g in goods if (a := _num(g.get("amount"))) is not None)
    return {
        "rows": matched,
        "unmatched_api": unmatched_api,
        "totals": {
            "pdf_sum": round(pdf_total, 2),
            "api_sum": round(api_total, 2),
            "status": _cmp_number(pdf_total, api_total)[0] if goods and products else "missing",
        },
    }


def compare_contract_data(
    extracted: dict, api_data: dict, validation: dict | None = None
) -> dict:
    """Yakuniy natija: { extracted, api, comparison, confidence }."""
    contract = api_data.get("contract") or api_data
    orientation, fields = _detect_orientation(extracted, contract)

    if validation:
        for path, entry in fields.items():
            fconf = validation["fields"].get(path, {}).get("confidence")
            if fconf:
                entry["ocr_confidence"] = fconf

    products_cmp = _compare_products(extracted.get("products") or [], contract)

    n_match = sum(1 for c in fields.values() if c["status"] == "match")
    n_mismatch = sum(1 for c in fields.values() if c["status"] == "mismatch")
    prod_mismatch = sum(
        1
        for row in products_cmp["rows"]
        for f in row.get("fields", {}).values()
        if f["status"] == "mismatch"
    ) + sum(1 for r in products_cmp["rows"] if r.get("status") == "unmatched_pdf")

    return {
        "extracted": extracted,
        "api": api_data,
        "comparison": {
            "orientation": orientation,  # import: Uz tomon = buyer
            "fields": fields,
            "products": products_cmp,
        },
        "confidence": {
            "matched": n_match,
            "mismatched": n_mismatch,
            "total_compared": len(fields),
            "product_issues": prod_mismatch,
            "validation": (validation or {}).get("summary"),
        },
    }

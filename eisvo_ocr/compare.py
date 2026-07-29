"""[7] compare_contract_data() — PDF'dan ajratilgan ma'lumot vs EISVO API JSON'i.

API strukturasi (real namunalar asosida): javob ildizida {"contract": {...}},
tomonlar contractorUz*/contractorFor*, tovarlar specifications[].goods[],
kodlar klassifikatorda (currCode "840", incotermsCode "08" ...) — tarjima codes.py da.

Import shartnomada Uz tomon = buyer; eksportda aksincha. Yo'nalish avtomatik
aniqlanadi (ikkala variant sinab, mosligi ko'prog'i olinadi).
"""
from __future__ import annotations

import re
import unicodedata
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
    ("{FOR}.address", "contractorForAddress", "address"),
    ("{FOR}.bank.name", "bankForName", "text"),
    # SWIFT + hisob API'da yagona erkin matn (bankForAttributes) — birlashtirib
    # tekshiriladi (pastdagi _bank_attributes_row), shuning uchun bu yerda alohida emas
    ("{UZ}.name", "contractorUzName", "text"),
    ("{UZ}.inn", "contractorUzInn", "id"),
    ("{UZ}.address", "contractorUzAddress", "address"),
    ("delivery.place", "deliveryTerms.0.destination", "text"),
    ("delivery.incoterms", "deliveryTerms.0.incotermsCode", "incoterms"),
]

# yo'nalishni aniqlashda ishlatiladigan maydonlar
_ORIENTATION_FIELDS = {"{FOR}.name", "{UZ}.name", "{UZ}.inn", "{FOR}.country"}

# Грузополучатель / Consignee — yo'nalishga bog'liq EMAS (API'da to'g'ridan-to'g'ri
# consignees[] massivida). Tovar turidagi shartnomalarda bor, quyidagi turlardan tashqari.
_RECEIVER_SPECS: list[tuple[str, str, str]] = [
    ("receiver.name", "consignees.0.name", "text"),
    ("receiver.country", "consignees.0.countryCode", "country"),
    ("receiver.inn", "consignees.0.inn", "id"),
]
_NO_RECEIVER_TYPES = {"10", "13", "22", "98", "99"}  # bu turlarda Грузополучатель yo'q

# yo'nalishdan MUSTAQIL qo'shimcha maydonlar (API'da to'g'ridan-to'g'ri yo'llarda):
# Грузоотправитель consignors[], Производитель manufacturers[], spetsifikatsiya, to'lov valyutasi/muddati
_EXTRA_SPECS: list[tuple[str, str, str]] = [
    ("supplier.name", "consignors.0.name", "text"),
    ("supplier.country", "consignors.0.countryCode", "country"),
    ("supplier.inn", "consignors.0.inn", "id"),
    ("manufacturer.name", "manufacturers.0.name", "text"),
    ("manufacturer.country", "manufacturers.0.countryCode", "country"),
    ("manufacturer.inn", "manufacturers.0.inn", "id"),
    ("specification_number", "specifications.0.docNo", "id"),
    ("payment_deadline", "importTerms.0.paymentDeadline", "id"),
    # payment_currency ATAYLAB yo'q: API accCurrCode1 doim bor va odatda shartnoma
    # valyutasi bilan bir xil — solishtirilsa soxta mismatch beradi (currency allaqachon
    # currCode1 bilan tekshiriladi). Maydon ajratiladi-yu, taqqoslanmaydi.
]


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
    # o'zbek/rus to'liq shakllar (translit + x->h fold'dan keyingi ko'rinishda):
    # «Хозяйственное Общество», «Hususiy Korxonasi/Korxona», «...jamiyati»
    "hozyaystvennoe", "obschestvo", "hususiy", "korhona", "korhonasi",
    "jamiyat", "jamiyati", "korxona", "korxonasi",
    # «Mas'uliyati Cheklangan Jamiyati» (MChJ = LLC) to'liq shakli — apostrof
    # olib tashlangach «masuliyati» bir so'z; rus «...ограниченной ответственностью»
    "masuliyati", "cheklangan", "ogranichennoy", "otvetstvennostyu",
    # «Aksiyadorlik Jamiyati» (AJ), ochiq/yopiq turlari
    "aksiyadorlik", "ochiq", "yopiq",
}

# kirill homoglif harflari (lotinga vizual mos) — ID'larda: «НАН» (kirill) == «HAH» (lotin).
# translit fonetik (Н->n) beradi, bu yerda esa VIZUAL moslik kerak (Н->H).
_HOMOGLYPH = str.maketrans({
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X", "І": "I", "Ј": "J",
    "а": "a", "в": "b", "е": "e", "к": "k", "м": "m", "н": "h", "о": "o",
    "р": "p", "с": "c", "т": "t", "у": "y", "х": "x",
})


def _strip_diacritics(s: str) -> str:
    """ý->y, é->e, ñ->n ... — aks holda [^a-z] filtri ularni bo'sh joyga aylantirib
    so'zni buzadi (Derýaplastik -> «der aplastik»)."""
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def _fold_word(w: str) -> str:
    """Bitta so'zni normalizatsiya kaliti shakliga (translit + diakritik + x->h)."""
    w = _strip_diacritics(_translit(str(w).lower())).replace("x", "h")
    return re.sub(r"[^0-9a-z]", "", w)


# joy nomlari gazetteeri: bir shaharning uz/ru/en variantlari (Buxoro=Buhara=Бухара,
# Toshkent=Tashkent) fuzzy bilan yutilmaydi (unli farqlari) — kanonik shaklga keltiramiz
_PLACE_CANON: dict[str, str] = {}
for _grp in [
    ["Buxoro", "Buhara", "Bukhara", "Бухара"],
    ["Toshkent", "Tashkent", "Ташкент"],
    ["Samarqand", "Samarkand", "Самарканд"],
    ["Andijon", "Andijan", "Андижан"],
    ["Namangan", "Наманган"],
    ["Navoiy", "Navoi", "Навои"],
    ["Urganch", "Urgench", "Ургенч"],
    ["Nukus", "Нукус"],
    ["Qarshi", "Karshi", "Карши"],
    ["Termiz", "Termez", "Термез"],
    ["Guliston", "Gulistan", "Гулистан"],
    ["Xiva", "Khiva", "Хива"],
]:
    _canon = _fold_word(_grp[0])
    for _name in _grp:
        _PLACE_CANON[_fold_word(_name)] = _canon


def _norm_text(s: Any) -> str:
    s = _strip_diacritics(_translit(str(s).lower()))
    # o'zbekcha x <-> h / rus х varianti (Buxoro<->Buhara, Suxrob<->Suhrob) — ikkala
    # tomon bir xil fold qilinadi, shuning uchun mavjud mosliklarni buzmaydi
    s = s.replace("x", "h")
    # so'z ICHIDAGI apostrofni olib tashlaymiz (bo'sh joyga emas) — aks holda
    # «mas'uliyati» -> «mas»+«uliyati» ikkiga bo'linadi va legal-form sifatida
    # tanilmaydi; «O'zbekiston» -> «ozbekiston» ham bir so'z bo'lib qoladi
    s = re.sub(r"[’'`ʻʼ´]", "", s)
    # qolgan tinish belgilarini (defis, qo'shtirnoq, nuqta...) bo'sh joyga
    s = re.sub(r"[^0-9a-z\s]", " ", s)
    # legal-form'larni tashla, joy nomlarini kanonik shaklga keltir
    return " ".join(_PLACE_CANON.get(t, t) for t in s.split() if t not in _LEGAL_FORMS)


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
    def norm(v: Any) -> str:
        v = str(v).translate(_HOMOGLYPH)  # kirill homoglifni lotinga (НАН->HAH)
        # № va # belgilarini HAR JOYDAN olib tashlaymiz («KEM-OZO №03/02» ichida ham),
        # bo'shliq/defisni ham — «KEM-OZO № 03/02» == «KEMOZO03/02»
        return re.sub(r"[\s\-–—№#]", "", v).upper()
    a, b = norm(ours), norm(theirs)
    return ("match" if a == b else "mismatch"), (1.0 if a == b else 0.0)


def _cmp_text(ours: Any, theirs: Any) -> tuple[str, float]:
    a, b = _norm_text(ours), _norm_text(theirs)
    # token_set ham: bir tomon ikkinchisining QISQARTMASI/subset bo'lsa (API ko'pincha
    # qisqaroq — «...Hususiy Korxonasi» vs «...XK», «г.Ашхабад, Туркменистан» vs «г.Ашхабад»)
    # token_sort past baho beradi, token_set esa to'g'ri.
    score = max(fuzz.token_sort_ratio(a, b), fuzz.token_set_ratio(a, b)) / 100
    ok = score >= settings.compare_fuzzy_threshold
    return ("match" if ok else "mismatch"), round(score, 3)


# manzillardagi umumiy/tur so'zlari — solishtirishda e'tiborga olinmaydi
# (har manzilda bor: viloyat/область, tuman/район, ko'cha, uy...). Faqat atoqli
# otlar (shahar/tuman/mahalla nomlari) taqqoslanadi.
_ADDR_GENERIC = {
    "tuman", "tumani", "rayon", "rayona", "rayoni", "viloyat", "viloyati",
    "oblast", "oblasti", "shahar", "shahri", "shaxar", "gorod", "goroda",
    "respublika", "respublikasi", "respubliki", "dom", "uyi", "ulitsa",
    "kocha", "kochasi", "mahalla", "mahallasi", "mfy", "mfj", "territoriya",
    "territoriyasi", "dvor", "street", "building", "poselok", "qishloq",
    "kishlak", "aholi", "punkti", "obl", "prospekt", "proezd",
    # ma'muriy/tavsif so'zlari (en/uz/ru) — proper ot emas, solishtirishda tashlanadi.
    # «Free Economic Zone» == «erkin iqtisodiy zonasi», «Massif» == «massivi»
    "region", "district", "city", "town", "village", "area", "zone", "zona",
    "zonasi", "free", "erkin", "ozod", "svobodnaya", "svobodnoy", "economic",
    "iqtisodiy", "ekonomicheskaya", "ekonomicheskoy", "hudud", "hududi",
    "hududiy", "massif", "massiv", "massivi", "massiva", "industrial", "sanoat",
    "promyshlennaya", "sez", "fez", "eiz", "road", "avenue", "block", "kvartal",
    "mikrorayon", "microrayon", "mfj", "house",
    # o'zbekiston/respublika o'zi — Uz tomon manzilida DOIM bor, farqlovchi emas
    "republic", "uzbekistan", "ozbekiston", "uzbekiston", "ozbekistan",
}

# rus toponimik SIFAT qo'shimchalari (translitdan keyin, lotinda) — «Сырдарьинская
# область», «Каршинский район», «Каганский район» kabi shakllarni o'zak toponimga
# keltirish uchun kesiladi: сырдарьинская->sirdar, каршинский->karsh, каганский->kagan.
# Shunda API'dagi o'zbekcha OT shakli (Sirdaryo, Qarshi, Kagan) bilan mos keladi.
_ADJ_SUFFIXES = (
    "inskaya", "inskoy", "inskiy", "evskaya", "evskiy", "ovskaya", "ovskiy",
    "skaya", "skoy", "skoe", "skie", "skogo", "skom", "skiy", "skij",
)


def _addr_tokens(s: Any) -> list[str]:
    """Manzildan atoqli ot tokenlari (>=4 harf, tur/umumiy so'z emas)."""
    return [t for t in _norm_text(s).split() if len(t) >= 4 and t not in _ADDR_GENERIC]


def _toponym_root(tok: str) -> str:
    """Toponim o'zagi: rus sifat qo'shimchasini kesib, q->k folding va gazetteer
    kanonini qo'llaydi. «каршинский»->karsh, «сырдарьинская»->sirdar, «qarshi»->qarshi
    (q->k->karshi->gazetteer->qarshi). Shu bilan sifat/ot va q/k tafovutlari yutiladi."""
    t = tok
    for suf in _ADJ_SUFFIXES:
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            t = t[: -len(suf)]
            break
    t = t.replace("q", "k")  # Qarshi==Karshi, Qashqadaryo==Kashkadaryo
    return _PLACE_CANON.get(t, t)


def _cmp_address(ours: Any, theirs: Any) -> tuple[str, float]:
    """Manzil: bir tomon (odatda API) qisqaroq — uning atoqli otlari ikkinchisida
    (fuzzy) uchrasa match. Toshkent↔Ташкент, tuman↔район farqlari shunda yutiladi.
    Har tokendan toponim O'ZAGI olinadi (rus sifat qo'shimchasi kesiladi), shuning
    uchun «Сырдарьинская»↔«Sirdaryo», «Каршинский»↔«Qarshi» ham mos keladi."""
    toks_a, toks_b = _addr_tokens(ours), _addr_tokens(theirs)
    if not toks_a or not toks_b:
        return _cmp_text(ours, theirs)  # atoqli ot yo'q — oddiy matn solishtiruvi
    roots_a = [_toponym_root(t) for t in toks_a]
    roots_b = [_toponym_root(t) for t in toks_b]
    # kamroq atoqli otga ega tomonni ikkinchisida qidiramiz (subset mantiqi):
    # har bir o'zak ikkinchi tomon o'zaklaridan birortasiga fuzzy mos kelsa — topildi
    (short, long_roots) = (roots_b, roots_a) if len(roots_b) <= len(roots_a) \
        else (roots_a, roots_b)
    found = sum(
        1 for s in short
        if s and any(l and fuzz.partial_ratio(s, l) >= 82 for l in long_roots)
    )
    score = found / len(short)
    return ("match" if score >= 0.6 else "mismatch"), round(score, 3)


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
    "address": _cmp_address,
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


def _compare_one(extracted: dict, contract: dict, our_path: str, api_path: str, kind: str) -> dict:
    ours = _get_path(extracted, our_path)
    theirs = _get_path(contract, api_path)
    ours_empty = ours is None or (isinstance(ours, str) and not ours.strip())
    theirs_empty = theirs is None or (isinstance(theirs, str) and not theirs.strip())
    if theirs_empty:
        # API'da qiymat yo'q — solishtirishga narsa yo'q (neytral)
        status, score = ("missing_api" if not ours_empty else "missing_pdf"), 0.0
    elif ours_empty:
        # API'da bor, lekin model PDF'dan chiqara olmagan → nomuvofiqlik
        status, score = "mismatch", 0.0
    else:
        status, score = _COMPARATORS[kind](ours, theirs)
    return {"pdf": ours, "api": theirs, "api_field": api_path, "status": status, "score": score}


def _compare_fields(extracted: dict, contract: dict, orientation: str) -> dict[str, dict]:
    for_side, uz_side = ("seller", "buyer") if orientation == "import" else ("buyer", "seller")
    out: dict[str, dict] = {}
    for our_tpl, api_path, kind in _FIELD_SPECS:
        our_path = our_tpl.replace("{FOR}", for_side).replace("{UZ}", uz_side)
        entry = _compare_one(extracted, contract, our_path, api_path, kind)
        entry["_tpl"] = our_tpl
        out[our_path] = entry
    out[f"{for_side}.bank.attributes"] = _bank_attributes_row(extracted, contract, for_side)
    # Грузополучатель / Consignee — yo'nalishdan mustaqil, faqat mos shartnoma turlarida
    if str(contract.get("cntrType") or "").strip().zfill(2) not in _NO_RECEIVER_TYPES:
        for our_path, api_path, kind in _RECEIVER_SPECS:
            entry = _compare_one(extracted, contract, our_path, api_path, kind)
            entry["_tpl"] = our_path
            out[our_path] = entry
    # Грузоотправитель / Производитель / spetsifikatsiya / to'lov — yo'nalishdan mustaqil
    for our_path, api_path, kind in _EXTRA_SPECS:
        entry = _compare_one(extracted, contract, our_path, api_path, kind)
        entry["_tpl"] = our_path
        out[our_path] = entry
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


def _key_tokens(name: str) -> set[str]:
    """Nomdagi model/o'lcham RAQAMLARI (8004, 50mm, 32212, 500) — mahsulotni
    farqlovchi belgilar. Katalogda nomlar deyarli bir xil, faqat shu raqamlar
    farq qiladi; ular umuman mos kelmasa — boshqa mahsulot."""
    return set(re.findall(r"[a-z]*\d+[a-z]*", _norm_text(name)))


# nom o'xshashligi shu chegaradan past bo'lsa, hs/qty tasodifiy mos kelsa ham
# juftlamaymiz (водонагреватель != фитинг, ikkalasi qty=500 bo'lsa ham)
_NAME_FLOOR = 0.45


def _compare_products(products: list[dict], contract: dict) -> dict:
    goods = [
        g
        for spec in contract.get("specifications") or []
        for g in spec.get("goods") or []
    ]
    matched, used = [], set()
    for p in products:
        pk = _key_tokens(p.get("name") or "")
        best_j, best_score = None, 0.0
        for j, g in enumerate(goods):
            if j in used:
                continue
            ns = _name_score(p.get("name") or "", g.get("itemsName") or "")
            if ns < _NAME_FLOOR:
                continue  # nomi umuman o'xshamaydi — juftlamaymiz
            gk = _key_tokens(g.get("itemsName") or "")
            if pk and gk and pk.isdisjoint(gk):
                continue  # model/o'lcham raqamlari ziddiyatli (8004 vs 8005) — boshqa mahsulot
            score = ns
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
    ) + sum(1 for r in products_cmp["rows"] if r.get("status") == "unmatched_pdf") \
      + len(products_cmp["unmatched_api"])  # API'da bor, PDF'da yo'q tovarlar ham nomuvofiqlik

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

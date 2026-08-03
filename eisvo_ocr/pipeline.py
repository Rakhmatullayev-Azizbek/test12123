"""Butun pipeline orkestri: PDF -> [1..7] -> natija."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from . import assembler, deterministic, llm_layer, pdf_utils, validator
from .codes import country_matches
from .compare import _cmp_text, compare_contract_data
from .schemas import ExtractedContract

log = logging.getLogger("eisvo_ocr")


def process_pdf(pdf_path: str | Path) -> dict:
    """PDF -> {extracted, validation, debug}. API taqqoslashsiz."""
    log.info("[1] PDF -> rasmlar: %s", pdf_path)
    images = pdf_utils.pdf_to_images(pdf_path)
    log.info("    %d sahifa", len(images))

    log.info("[2] Surya (layout + table + text + reading order)")
    from .surya_layer import run_surya  # lazy: GPU modellari shu yerda yuklanadi

    ocr_doc = run_surya(images)

    log.info("[3] Assembler: markdown + jadval davomlarini ulash")
    assembled = assembler.assemble(ocr_doc)
    log.info("    %d jadval, %d belgi matn", len(assembled.tables), len(assembled.markdown))

    log.info("[4] Deterministik extraction (regex + normalizatsiya)")
    hints = deterministic.extract_candidates(assembled.plain_text)

    log.info("[5A] LLM META")
    meta = llm_layer.extract_meta(assembled.markdown, hints)
    log.info("[5B] LLM PRODUCTS (%d jadval aniqlangan; tovarlar butun matndan ajratiladi)",
             len(assembled.tables))
    products = llm_layer.extract_products(assembled.tables_markdown, assembled.markdown)

    extracted = ExtractedContract(meta=meta, products=products)

    # LLM natijasini deterministik normalizatsiya bilan tozalaymiz
    if extracted.meta.contract_date:
        extracted.meta.contract_date = (
            deterministic.normalize_date(extracted.meta.contract_date)
            or extracted.meta.contract_date
        )
    if extracted.meta.delivery and extracted.meta.delivery.incoterms:
        extracted.meta.delivery.incoterms = (
            deterministic.normalize_incoterms(extracted.meta.delivery.incoterms)
            or extracted.meta.delivery.incoterms
        )

    # 9-xonali INN = O'zbek STIR — model ba'zan uni chet el tomoniga biriktiradi;
    # O'zbek tomonga qaytaramiz (uchinchi shaxs default'idan OLDIN, receiver INN'ni olsin).
    _reassign_uzbek_inn(extracted.meta)

    # Uchinchi shaxs yorliqlari matnda topilmasa (OCR buzgan yoki umuman yo'q),
    # nom/mamlakat/INN'ini mos asosiy tomondan meros qilib olamiz.
    _fill_third_party_defaults(extracted.meta)

    log.info("[6] Validator")
    validation = validator.validate(extracted, assembled.plain_text)
    log.info("    natija: %s", validation["summary"])

    return {
        "extracted": extracted.model_dump(),
        "validation": validation,
        "debug": {
            "hints": hints,
            "markdown": assembled.markdown,
            "tables_markdown": assembled.tables_markdown,
        },
    }


def compare_with_api(result: dict, api_data: dict) -> dict:
    """[7] Tayyor process_pdf natijasini API JSON'i bilan taqqoslash."""
    flat = _flatten_extracted(result["extracted"])
    return compare_contract_data(flat, api_data, validation=result["validation"])


def process_and_compare(pdf_path: str | Path, api_data: dict) -> dict:
    """To'liq oqim: PDF -> extraction -> API bilan taqqoslash."""
    return compare_with_api(process_pdf(pdf_path), api_data)


# Uchinchi shaxs yorlig'i matnda bo'lmaganda qaysi asosiy tomondan meros olinishi:
#   Грузополучатель (receiver)   ≈ xaridor (buyer)   — tovarni qabul qiluvchi
#   Грузоотправитель (supplier)  ≈ sotuvchi (seller) — tovarni jo'natuvchi
#   Производитель (manufacturer)  ≈ sotuvchi (seller) — odatda ishlab chiqaruvchi
# (EISVO ham aynan shu qoida bilan avtomatik to'ldiradi: consignee=buyer,
# consignor=seller, manufacturer=seller — shuning uchun default mos keladi.)
_THIRD_PARTY_INN_FALLBACK = {
    "receiver": "buyer",
    "supplier": "seller",
    "manufacturer": "seller",
}


def _is_foreign_party(party) -> bool:
    """Tomon aniq CHET EL (mamlakati ko'rsatilgan va O'zbekiston EMAS)."""
    c = (getattr(party, "country", None) or "").strip()
    return bool(c) and country_matches(c, "860") is False


def _reassign_uzbek_inn(meta) -> None:
    """9-xonali INN — O'zbek STIR formati (RE_INN). Model uni ba'zan CHET EL tomoniga
    (masalan xitoylik sotuvchiga) biriktirib qo'yadi. Chet el tomon 9-xonali INNga ega
    bo'lolmaydi; agar ikkinchi (O'zbek) tomonda INN bo'lmasa — o'sha tomonga ko'chiramiz.
    Faqat O'zbek tomonda INN bo'sh bo'lsagina ishlaydi (mavjud INN ustidan yozmaymiz)."""
    for foreign, other in ((meta.seller, meta.buyer), (meta.buyer, meta.seller)):
        if foreign is None or not _is_foreign_party(foreign):
            continue
        inn = re.sub(r"\s", "", foreign.inn or "")
        if not re.fullmatch(r"\d{9}", inn):
            continue
        # chet el kompaniyasida O'zbek STIR bo'lolmaydi -> chet el tomondan olib tashlaymiz
        foreign.inn = None
        # O'zbek tomonda INN bo'lmasa -> o'sha tomonga ko'chiramiz (mavjudini o'zgartirmaymiz)
        if other is not None and not (other.inn and other.inn.strip()):
            other.inn = inn


def _same_company(a, b) -> bool:
    """Ikki nom bir tashkilotni bildiradimi (translit + legal-form + fuzzy)."""
    if not (a and str(a).strip() and b and str(b).strip()):
        return False
    status, _ = _cmp_text(a, b)
    return status == "match"


def _fill_third_party_defaults(meta) -> None:
    """Uchinchi shaxs (receiver/supplier/manufacturer) INN/nom/mamlakatini mos asosiy
    tomondan (buyer/seller) meros qilib olamiz. Ikki holat:

    1) Uchinchi shaxs UMUMAN yorliqsiz (na nom, na INN) — matnda topilmagan; nom,
       mamlakat va INN'ni to'liq meros olamiz.
    2) Nom BOR, lekin INN yo'q — ko'p shartnomada Грузополучатель buyer bilan bir xil
       nom bilan yozilib, INN takrorlanmaydi. Nom asosiy tomon bilan mos kelsa,
       INN'ni o'shandan to'ldiramiz (EISVO ham consignee.inn'ni buyer'dan avto-oladi).
       Nom farq qilsa — boshqa tashkilot, INN'ni KO'CHIRMAYMIZ.
    """
    for tp_name, main_name in _THIRD_PARTY_INN_FALLBACK.items():
        tp = getattr(meta, tp_name, None)
        main = getattr(meta, main_name, None)
        if tp is None or main is None:
            continue
        has_name = bool(tp.name and tp.name.strip())
        has_inn = bool(tp.inn and tp.inn.strip())
        main_inn = (main.inn or "").strip()
        if has_inn:
            continue
        if not has_name:
            # Holat 1: to'liq yorliqsiz — nom/mamlakat/INN'ni meros olamiz
            if main.name and main.name.strip():
                tp.name = main.name.strip()
            main_country = getattr(main, "country", None)
            if main_country and str(main_country).strip():
                tp.country = main_country
            if main_inn:
                tp.inn = main_inn
        elif main_inn and _same_company(tp.name, main.name):
            # Holat 2: nom bor, INN yo'q, nom asosiy tomon bilan bir xil — INN'ni olamiz
            tp.inn = main_inn


def _flatten_extracted(extracted: dict) -> dict:
    """{meta: {...}, products: {...}} -> compare.FIELD_MAP kutadigan tekis struktura."""
    meta = extracted.get("meta", {})
    return {
        **meta,
        "products": extracted.get("products", {}).get("products", []),
        "products_total": extracted.get("products", {}).get("total_amount"),
    }

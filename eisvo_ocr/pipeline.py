"""Butun pipeline orkestri: PDF -> [1..7] -> natija."""
from __future__ import annotations

import logging
from pathlib import Path

from . import assembler, deterministic, llm_layer, pdf_utils, validator
from .compare import compare_contract_data
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
    log.info("[5B] LLM PRODUCTS")
    products = llm_layer.extract_products(assembled.tables_markdown)

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


def _flatten_extracted(extracted: dict) -> dict:
    """{meta: {...}, products: {...}} -> compare.FIELD_MAP kutadigan tekis struktura."""
    meta = extracted.get("meta", {})
    return {
        **meta,
        "products": extracted.get("products", {}).get("products", []),
        "products_total": extracted.get("products", {}).get("total_amount"),
    }

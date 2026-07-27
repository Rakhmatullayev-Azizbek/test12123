"""EISVO LLM OCR — tashqi savdo shartnomalarini PDF'dan strukturaviy ajratish.

Pipeline: PDF -> Surya OCR -> Assembler -> Deterministik -> LLM (Qwen3) -> Validator -> Compare.
"""
from .compare import compare_contract_data
from .pipeline import process_and_compare, process_pdf

__all__ = ["process_pdf", "process_and_compare", "compare_contract_data"]

"""[5] LLM QATLAMI — Qwen3-14B AWQ (vLLM, guided_json).

vLLM Windows'da ishlamaydi — server WSL2 yoki alohida Linux mashinada turadi:
  vllm serve Qwen/Qwen3-14B-AWQ --max-model-len 16384 --gpu-memory-utilization 0.9
Bu modul unga OpenAI-mos HTTP klient orqali ulanadi (config.vllm_base_url).

Bosqich A: META (seller/buyer/bank/delivery) — butun matn.
Bosqich B: PRODUCTS — faqat jadval markdown'lari.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from openai import OpenAI
from pydantic import BaseModel

from .config import settings
from .schemas import ContractMeta, ProductList

_SYSTEM = (
    "Sen tashqi savdo shartnomalaridan strukturaviy ma'lumot ajratuvchi yordamchisan. "
    "Faqat berilgan matndagi faktlarni chiqar. Matnda yo'q qiymatni O'YLAB TOPMA — null qoldir. "
    "«so'z⟨0.62⟩» ko'rinishidagi belgi OCR ishonchi pastligini bildiradi: qiymatni "
    "kontekstga qarab ehtiyotkorlik bilan o'qi, ⟨...⟩ belgining o'zini natijaga KIRITMA. "
    "Sanalarni YYYY-MM-DD formatida, summalarni son (float) sifatida qaytar."
)

_META_PROMPT = """Quyida shartnomaning to'liq matni (OCR) va deterministik qidiruv topgan nomzodlar berilgan.
Shartnoma meta-ma'lumotlarini JSON sxema bo'yicha ajrat.

Alohida e'tibor ber (bular shartnomalarda deyarli doim bor):
- valyuta (currency) va umumiy summa (total_amount) — «Валюта контракта», «Общая сумма» kabi joylarda
- har ikki tomonning BANK rekvizitlari: bank nomi, MFO, SWIFT, hisob raqami («Банк продавца/покупателя», «р/с»)
- yetkazib berish sharti va JOYI («Условия поставки: CPT Ташкент» -> incoterms=CPT, place=Ташкент)

Deterministik nomzodlar (regex bilan topilgan, yordam uchun — lekin matnga mos kelishini tekshir):
{hints}

=== SHARTNOMA MATNI ===
{text}
"""

_PRODUCTS_PROMPT = """Quyida shartnomadagi tovarlar jadval(lar)i markdown ko'rinishida berilgan.
Har bir tovar qatorini JSON sxema bo'yicha ajrat. Jami/итого qatorini products ro'yxatiga
QO'SHMA — uni total_amount maydoniga yoz.

=== JADVALLAR ===
{tables}
"""


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(
        base_url=settings.vllm_base_url,
        api_key=settings.vllm_api_key,
        timeout=settings.llm_timeout,
    )


def _make_strict(node) -> None:
    """Barcha maydonlarni required qilish (OpenAI strict uslubi) — model har bir
    maydonni chiqarishga majbur bo'ladi (qiymat topilmasa null), tashlab ketolmaydi."""
    if isinstance(node, dict):
        if node.get("type") == "object" and isinstance(node.get("properties"), dict):
            node["required"] = list(node["properties"].keys())
            node["additionalProperties"] = False
        for v in node.values():
            _make_strict(v)
    elif isinstance(node, list):
        for v in node:
            _make_strict(v)


def _truncate(text: str) -> str:
    """Juda uzun hujjatlar kontekstga sig'maydi — oxiridan kesamiz (rekvizitlar
    odatda hujjat boshida bo'ladi)."""
    limit = settings.llm_max_input_chars
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[... matn qisqartirildi: hujjat juda uzun ...]"


def _guided_call(prompt: str, schema_model: type[BaseModel]) -> BaseModel:
    schema = schema_model.model_json_schema()
    _make_strict(schema)
    # javob limitini kontekstga moslash: kirill matnida ~2 belgi/token deb baholaymiz
    est_input_tokens = (len(_SYSTEM) + len(prompt)) // 2 + 500
    max_out = max(1024, min(settings.llm_max_tokens,
                            settings.llm_context_len - est_input_tokens))
    resp = _client().chat.completions.create(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=max_out,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
        # vLLM >=0.25: eski `guided_json` extra parametri indamay e'tiborga
        # olinmaydi — strukturaviy chiqish faqat response_format orqali ishlaydi
        response_format={
            "type": "json_schema",
            "json_schema": {"name": schema_model.__name__, "schema": schema},
        },
        extra_body={
            # Qwen3: extraction uchun thinking shart emas — tezlik uchun o'chiramiz
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    raw = resp.choices[0].message.content or "{}"
    return schema_model.model_validate(json.loads(raw))


def extract_meta(full_text: str, hints: dict) -> ContractMeta:
    """Bosqich A."""
    prompt = _META_PROMPT.format(
        hints=json.dumps(hints, ensure_ascii=False, indent=1),
        text=_truncate(full_text),
    )
    return _guided_call(prompt, ContractMeta)  # type: ignore[return-value]


_RE_TOTAL_ROW = re.compile(r"итого|всего|jami|жами|total\b", re.IGNORECASE)


def _drop_total_rows(pl: ProductList) -> ProductList:
    """Prompt taqiqlasa ham model ba'zan Итого qatorini tovar qilib qo'shadi —
    deterministik olib tashlaymiz (summasi total_amount'ga o'tadi)."""
    kept = []
    for p in pl.products:
        name = (p.name or "").strip()
        if _RE_TOTAL_ROW.search(name) and (p.quantity is None or p.price is None):
            if pl.total_amount is None and p.amount is not None:
                pl.total_amount = p.amount
            continue
        kept.append(p)
    pl.products = kept
    return pl


def extract_products(tables_markdown: str) -> ProductList:
    """Bosqich B."""
    if not tables_markdown.strip():
        return ProductList()
    prompt = _PRODUCTS_PROMPT.format(tables=_truncate(tables_markdown))
    result = _guided_call(prompt, ProductList)  # type: ignore[return-value]
    return _drop_total_rows(result)  # type: ignore[arg-type]

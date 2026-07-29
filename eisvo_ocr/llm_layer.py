"""[5] LLM QATLAMI — Qwen3-14B AWQ (vLLM, guided_json).

vLLM Windows'da ishlamaydi — server WSL2 yoki alohida Linux mashinada turadi:
  vllm serve Qwen/Qwen3-14B-AWQ --max-model-len 16384 --gpu-memory-utilization 0.9
Bu modul unga OpenAI-mos HTTP klient orqali ulanadi (config.vllm_base_url).

Bosqich A: META (seller/buyer/bank/delivery) — butun matn.
Bosqich B: PRODUCTS — faqat jadval markdown'lari.
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache

from openai import OpenAI
from pydantic import BaseModel

from .config import settings
from .schemas import ContractMeta, ProductList

log = logging.getLogger("eisvo_ocr")

_SYSTEM = (
    "You extract structured data from foreign-trade contracts. Output ONLY facts present "
    "in the given text. NEVER invent a value that is not in the text — leave it null. "
    "A marker like «word⟨0.62⟩» means low OCR confidence for that word: read it carefully "
    "in context and NEVER include the ⟨...⟩ marker itself in your output. Copy values "
    "verbatim; do not translate or transliterate. Return dates as YYYY-MM-DD and amounts "
    "as numbers (float)."
)

_META_PROMPT = """Extract contract METADATA from the OCR text below into the given JSON schema.
Do NOT extract product/table rows here — products are handled separately.

LANGUAGE: values may be in English, Russian, or Uzbek — all three are valid sources; copy
them verbatim (do NOT translate or transliterate). If the same field appears in several,
prefer English > Uzbek > Russian. A fully-Uzbek requisites block is normal — extract every
field of it. Values printed ONLY in another language (Chinese, Korean, Turkish...) are not
copied. Codes/numbers (INN, MFO, account, SWIFT, dates, amounts) are language-neutral —
always extract them.

PARTIES: exactly one SELLER (Продавец/Поставщик/Sotuvchi/Supplier/Seller) and one BUYER
(Покупатель/Xaridor/Buyer/Customer). A party label is a HARD BOUNDARY: every requisite (name,
address, country, INN, bank name/address/MFO/account/SWIFT, director) after a label belongs to
that party until the next party label. Assign ownership by LABEL — never by page side or
column. Third-party labels (Грузополучатель/Consignee, Грузоотправитель/Consignor,
Производитель/Manufacturer) also start their own block and must not leak into seller/buyer.

INN vs BANK INN: a bare «INN/ИНН/STIR:» line inside a party's requisites is the COMPANY's inn
(even when it sits between the MFO and the account number). Treat it as the bank's only if it
is labeled «Bank INN/ИНН банка» or attached in parentheses to the bank name.

BANK: MFO/МФО -> bank.mfo; Р/с/Счет/Sch, or a currency-prefixed grouped number («USD-3232 3232
...») -> bank.account (copy exactly, keep the grouping). «Beneficiary/Банк получателя» = the
SELLER's bank; «Correspondent/Intermediary/Банк-корреспондент» is routing only — ignore it.
Address: prefer the legal address over the actual one. Seller and buyer are DIFFERENT companies
with DIFFERENT banks — if one account/SWIFT lands under both, you mis-assigned a block; fix it.

contract_number: the FULL contract/document identifier as printed, INCLUDING any letter
prefix that is part of it — «КОНТРАКТ КЕМ-ОZО № 03/02» -> «KEM-OZO 03/02», «XCUZB-E4006P»,
«ZAP-0312». Do NOT keep only the trailing digits after «№».

total_amount: ONLY the contract price clause («Общая сумма контракта ...», «Shartnoma umumiy
summasi ...»). Do NOT take the ИТОГО/Total/Jami footer under a goods table (that is one
specification's total, not the contract's). If only a table footer total is present, leave
total_amount null.

payment_terms: the payment METHOD only (prepayment/advance/letter of credit/CAD/collection/open
account) — never day or month counts.

delivery: incoterms = the trade CODE (EXW/FCA/FOB/CFR/CIF/CPT/CIP/DAP/DPU/DDP...), never the
edition year («Incoterms 2020»). place = the location in the Incoterms basis clause («Условия
поставки: CIP Ташкент» -> incoterms=CIP, place=Ташкент), NOT a street-level consignee address.

THIRD PARTIES — fill each ONLY when its explicit label is present (do NOT guess; it may equal
a main party, that is fine): receiver = Грузополучатель/Consignee/Получатель/Qabul qiluvchi;
supplier = Грузоотправитель/Consignor/Shipper; manufacturer = Производитель/Изготовитель/
Manufacturer. For each, fill name/country/address/inn. If no label, leave that party null —
never copy the buyer's/seller's name into it.

specification_number = the specification/appendix identifier ONLY («Спецификация №1» -> «1»,
«Приложение №2» -> «2»), never the word or the whole heading.
payment_currency = the payment currency code ONLY if it differs from the contract currency.
payment_deadline = the payment period as a number of days/months only («180 days») — never the
method. contract_type = the contract type if it is explicitly stated.

SWIFT check: characters 5-6 of a SWIFT are the bank's country code and must match that bank's
country. If the seller/buyer SWIFTs are swapped, swap them back; otherwise keep as printed.

Regex candidates (hints only — verify each against the text):
{hints}

=== CONTRACT TEXT ===
{text}
"""

_PRODUCTS_PROMPT = """Extract the product/goods table rows from the contract text below into the
given JSON schema. Fields per row: name, hs_code, unit, quantity, price (per unit), amount
(row total).

LANGUAGE: product cells may be Russian, Uzbek, or English — copy verbatim, do not translate.
Codes and numbers (HS code, quantity, price, amount) are language-neutral — always extract them.

BILINGUAL NAME = ONE PRODUCT: the SAME item is often written in two languages (Russian +
English), frequently on separate lines — e.g. «24/1-100% хлопчатобумажная кардная пряжа,
ткацкая» followed by «24/1-100% cotton yarn carded for weaving». This is ONE product whose
quantity/price/amount are stated ONCE. Output a SINGLE product for it (you may keep both
language names joined). NEVER output it as two products, and NEVER add the two language
versions' amounts together.

EXTRACT EVERY data row as one product — do not summarize or skip rows. SKIP header rows and
total/subtotal rows (ИТОГО/Всего/TOTAL/Jami): never add them as products (their sum belongs to
total_amount, not to a product).

ROW INDEX: rows are often prefixed with a running number («1.», «2)», ...). It is NOT data —
strip the leading «N.»/«N)» from the name, and never treat it as hs_code/quantity/price. Use it
to (a) bind each name to the numbers on the SAME visual row when columns are misaligned, and
(b) check completeness: indices should run consecutively with no gaps — the highest index is the
expected row count. Re-scan if your row count differs.

hs_code vs article/part number — DO NOT CONFUSE: hs_code (ТН ВЭД / HS customs code) is ALWAYS
pure digits (usually 8-10, e.g. «3204170000»). A manufacturer's article/catalog/model number
(«Артикул»/«Каталожный номер»/«Part No», e.g. «CVR FVR», «ABC-123») contains LETTERS and is NOT
an hs_code. Put only the pure-numeric customs code in hs_code; if a row has only a
letter-containing article number and no numeric code, leave hs_code null — never put the
article number there.

UNIT/CURRENCY DECLARED IN A HEADER: often the unit has no column of its own but is stated once
in another column's header, e.g. «Кол-во, кг» / «Вес нетто (кг)» (quantity is in kilograms) or
«Цена USD/кг» / «Total in USD». In that case use it as the unit for every row (unit = «kg»).
Prices and amounts are numeric only — strip currency symbols ($, €, £).

DO NOT INVENT: if a column does not exist in the table, leave that field null for every row —
never fabricate a plausible code/unit/quantity. Never extract the same product twice.

=== CONTRACT TEXT / TABLE ===
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
    finish = resp.choices[0].finish_reason
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # javob token limitiga yetib o'rtada kesilgan bo'lishi mumkin (finish='length')
        # — butun pipeline yiqilmasin: kesilishdan oldingi TO'LIQ obyektlarni tiklaymiz
        objs = _recover_objects(raw)
        log.warning("LLM JSON to'liq emas (finish=%s) — %d ta to'liq obyekt tiklandi",
                    finish, len(objs))
        data = {"products": objs}  # amalda faqat PRODUCTS bosqichi shuncha uzun chiqadi
    return schema_model.model_validate(data)


def _recover_objects(raw: str) -> list[dict]:
    """Kesilgan JSON matnidan to'liq `{...}` obyektlarni ketma-ket ajratib olamiz
    (tashqi o'ram va tugallanmagan oxirgi obyekt e'tiborga olinmaydi)."""
    dec = json.JSONDecoder()
    objs: list[dict] = []
    i, n = 0, len(raw)
    while i < n:
        if raw[i] != "{":
            i += 1
            continue
        try:
            obj, end = dec.raw_decode(raw, i)
        except json.JSONDecodeError:
            i += 1
            continue
        # faqat tovar obyektlari (nom maydoni bor) — tashqi {"products": ...} o'rami emas
        if isinstance(obj, dict) and "name" in obj:
            objs.append(obj)
        i = end
    return objs


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


def _model_nums(name: str | None) -> set[str]:
    """Nomdagi raqam ketma-ketliklari (24, 1, 100, 8004) — mahsulotni farqlovchi
    belgilar; til-neytral (translit shart emas)."""
    return set(re.findall(r"\d+", (name or "").lower()))


def _cyr_share(name: str | None) -> float:
    """Nomdagi harflarning necha ulushi kirill ekani (0..1)."""
    letters = [c for c in (name or "") if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "Ѐ" <= c <= "ӿ") / len(letters)


def _cross_script(a: str | None, b: str | None) -> bool:
    """Ikki nom TURLI skriptdami (biri asosan kirill, ikkinchisi asosan lotin)?
    Chin ikki tilli nusxa (rus + ingliz) doim shunday bo'ladi. Bu esa «правый» va
    «левый» kabi BIR tildagi, narxi tasodifan bir xil ikki alohida tovarni xato
    birlashtirishning oldini oladi — ular bir xil skriptda."""
    return (_cyr_share(a) >= 0.5) != (_cyr_share(b) >= 0.5)


def _merge_bilingual(pl: ProductList) -> ProductList:
    """Bir mahsulot ikki tilda (rus + ingliz) yozilib, model IKKI marta chiqargan bo'lsa
    birlashtiramiz. Shart: nomlar TURLI skriptda (ikki tilli nusxa belgisi) VA bir xil
    (hs, miqdor, narx, summa) YOKI ikkinchisi raqamsiz nusxa, VA nomdagi model-raqamlari
    ziddiyatli emas (8004 vs 8005 — boshqa mahsulot, birlashtirilmaydi).

    Skript sharti muhim: «Кронштейн ... правый» va «Кронштейн ... левый» bir tilda, narxi/
    miqdori tasodifan bir xil IKKI ALOHIDA tovar — cross-script bo'lmagani uchun endi
    birlashmaydi (avval xato birlashib, bittasi yo'qolardi)."""
    kept: list = []
    for p in pl.products:
        pn = _model_nums(p.name)
        target = None
        for q in kept:
            qn = _model_nums(q.name)
            # nomdagi model raqamlari to'liq TENG bo'lishi shart. «800/2100» va «900/2100»
            # umumiy «2100»ni ulashadi, lekin TENG emas -> turli mahsulot (birlashtirilmaydi);
            # ikki tilli nusxada esa raqamlar aynan bir xil (24/1-100% == 24/1-100%).
            if pn != qn:
                continue
            if not _cross_script(p.name, q.name):
                continue  # bir xil skript -> ikki tilli nusxa emas, alohida tovar
            p_empty = p.quantity is None and p.price is None and p.amount is None
            same_nums = (
                str(p.hs_code or "") == str(q.hs_code or "")
                and p.quantity == q.quantity and p.price == q.price and p.amount == q.amount
            )
            if p_empty or same_nums:
                target = q
                break
        if target is None:
            kept.append(p)
            continue
        if p.name and target.name and p.name not in target.name:
            target.name = f"{target.name} / {p.name}"  # ikki tildagi nomni saqlaymiz
    pl.products = kept
    return pl


def extract_products(tables_markdown: str, full_markdown: str = "") -> ProductList:
    """Bosqich B. Surya jadval aniqlashi ISHONCHSIZ: chiziqsiz/bo'sh formatli tovar
    ro'yxatini jadval deb tanimaydi, o'rniga boshqa kichik jadvalni (rekvizit qutisi)
    topib qo'yishi mumkin — u holda tables_markdown bo'sh bo'lmasa ham tovarlar yo'q.
    Shuning uchun DOIM butun hujjat markdown'idan ajratamiz: u aniqlangan jadvalni
    (| ... | ko'rinishida) ham, jadval deb tan olinmagan tovar qatorlarini ham o'z
    ichiga oladi. Faqat markdown bo'sh bo'lsagina tables_markdown'ga qaytamiz."""
    source = full_markdown.strip() or tables_markdown.strip()
    if not source:
        return ProductList()
    prompt = _PRODUCTS_PROMPT.format(tables=_truncate(source))
    result = _guided_call(prompt, ProductList)  # type: ignore[return-value]
    return _merge_bilingual(_drop_total_rows(result))  # type: ignore[arg-type]


# EISVO LLM OCR

Tashqi savdo shartnomalari (PDF) dan strukturaviy ma'lumot ajratish va EISVO API ma'lumoti bilan taqqoslash.

## Arxitektura

```
PDF → [1] pdf_to_images (PyMuPDF, 200 DPI)
    → [2] Surya (layout + table rec + text rec + reading order, batch)
    → [3] Assembler (markdown, jadval davomlarini ulash, low-conf belgilash)
    → [4] Deterministik extraction (regex: INN/MFO/hisob/SWIFT/sana; normalizatsiya)
    → [5] LLM — Qwen3-14B AWQ (vLLM guided_json): A) META, B) PRODUCTS
    → [6] Validator (arifmetika, format, OCR-match, per-field confidence)
    → [7] compare_contract_data() → {extracted, api, comparison, confidence}
```

| Modul | Bosqich |
|---|---|
| `eisvo_ocr/pdf_utils.py` | [1] PDF → rasmlar |
| `eisvo_ocr/surya_layer.py` | [2] Surya (barcha surya chaqiruvlari shu yerda izolyatsiya qilingan) |
| `eisvo_ocr/assembler.py` | [3] Document assembler |
| `eisvo_ocr/deterministic.py` | [4] Regex + normalizatsiya |
| `eisvo_ocr/schemas.py`, `llm_layer.py` | [5] LLM (guided_json sxemalar) |
| `eisvo_ocr/validator.py` | [6] Veto + confidence |
| `eisvo_ocr/compare.py` | [7] API bilan taqqoslash |
| `eisvo_ocr/pipeline.py` | Orkestr |

## O'rnatish (Docker — tavsiya etiladi)

Windows'da ikki to'siq bor: (1) vLLM Windows'da ishlamaydi, (2) Smart App Control
PyTorch DLL'larini bloklaydi (xato 4551). Docker (WSL2 backend) ikkalasini ham hal qiladi —
GPU passthrough ishlaydi, SAC konteynerga ta'sir qilmaydi.

```powershell
docker compose build             # pipeline image (torch+surya)
docker compose up -d web         # hamma kerakli servislarni ko'taradi:
                                 #   surya (OCR modeli), vllm (Qwen3), web (FastAPI)
```

Birinchi ishga tushishda modellar yuklab olinadi (Qwen3 ~10GB, Surya ~4GB) — 
`docker compose logs -f surya vllm` bilan kuzatish mumkin.

## Ishlatish

### Web interfeys (asosiy yo'l)

`http://localhost:8080` — PDF (majburiy) + EISVO API JSON (ixtiyoriy) yuklab
«Tekshirish» bosiladi. Natija: maydonlar taqqoslashi, tovarlar jadvali,
validator ogohlantirishlari, OCR markdown.

Backend: FastAPI (`eisvo_ocr/server.py`), endpoint: `POST /api/process`
(multipart: `pdf`, `api_json`). Frontend: `eisvo_ocr/static/index.html` (vanilla JS).

### CLI

```powershell
# PDF'ni data/ papkaga qo'ying, keyin:
docker compose run --rm app /data/contract.pdf -o /data/result.json
docker compose run --rm app /data/contract.pdf --api /data/api.json -o /data/result.json

# Surya bosqichlarini LLM'siz tekshirish:
docker compose run --rm --entrypoint python app /data/e2e_test.py
```

> **GPU taqsimoti**: bitta 24GB kartada vLLM `--gpu-memory-utilization 0.62`
> (~15GB) bilan ishlaydi, qolgan ~9GB Surya'ga yetadi. vLLM alohida serverda
> bo'lsa, `EISVO_VLLM_BASE_URL` ni o'sha manzilga qo'ying va limitni oshiring.

### Docker'siz (agar SAC o'chirilgan bo'lsa)

```powershell
.venv\Scripts\pip install -r requirements.txt   # Linux'da torch avto-CUDA; Windows'da cu130 wheel kerak
python main.py contract.pdf -o result.json
```

## TODO (real ma'lumot kelganda)

- [ ] `compare.py` dagi `FIELD_MAP` — real EISVO API javob sxemasi bo'yicha to'ldirish
- [ ] Namuna PDF'larda surya versiya API mosligini tekshirish (`surya_layer.py`)
- [ ] `schemas.py` maydonlarini real shartnoma maydonlariga moslash

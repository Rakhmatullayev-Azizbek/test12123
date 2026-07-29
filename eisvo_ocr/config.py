"""Markaziy sozlamalar. Har bir qiymatni EISVO_ prefiksli env var bilan
o'zgartirish mumkin, masalan: EISVO_VLLM_BASE_URL=http://gpu-server:8000/v1
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EISVO_", env_file=".env", extra="ignore")

    # [1] PDF -> images
    pdf_dpi: int = 200

    # [2] Surya
    surya_batch_size: int | None = None  # None = surya o'zi tanlaydi
    device: str = "cuda"

    # [3] Assembler
    low_conf_threshold: float = 0.70          # so'zni «so'z⟨0.62⟩» deb belgilash chegarasi
    table_merge_col_tolerance: float = 0.05   # ustun x-markazlari O'RTACHA farqi (sahifa eniga nisbatan)

    # [5] LLM — vLLM OpenAI-mos server (Windows'da vLLM yo'q: WSL2 yoki remote server)
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "EMPTY"
    llm_model: str = "Qwen/Qwen3-14B-AWQ"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 8192            # javob uchun; kontekstga sig'masa avto-kamayadi
                                          # (ko'p tovarli jadvallar uzun JSON beradi)
    llm_context_len: int = 20480          # server --max-model-len bilan bir xil bo'lsin
    llm_max_input_chars: int = 55_000     # juda uzun hujjatlar uchun himoya chegarasi (~16k token)
    llm_timeout: float = 600.0
    llm_stream: bool = True               # javobni oqim (streaming) bilan olish — masofaviy
                                          # proksi (Cloudflare ~100s) uzun javobni uzmasligi uchun

    # [6] Validator
    arithmetic_rel_tolerance: float = 0.01    # 1% — OCR yaxlitlash xatolari uchun
    ocr_match_min_ratio: float = 0.85         # LLM qiymati OCR matnida bormi (fuzzy)

    # [7] Comparison
    compare_fuzzy_threshold: float = 0.85

    # EISVO API — ID orqali shartnoma JSON'ini olish (JSON yuklash o'rniga).
    # POST so'rovi: body = (required maydonlar) + {id_field: kiritilgan ID}, Bearer token bilan.
    eisvo_api_url: str = ""             # POST endpoint to'liq URL (.env: EISVO_API_URL)
    eisvo_api_token: str = ""           # Bearer token (.env: EISVO_API_TOKEN) — muddати tugasa yangilanadi
    eisvo_api_id_field: str = "idn"     # so'rov body'sida ID qaysi kalit bilan yuboriladi
    eisvo_api_body: str = ""            # boshqa REQUIRED maydonlar — inline JSON (ixtiyoriy)
    eisvo_api_body_file: str = ""       # ...yoki JSON fayl yo'li (mas. /data/api_body.json) — afzal
    eisvo_api_timeout: float = 30.0


settings = Settings()

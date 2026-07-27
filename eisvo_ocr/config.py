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
    llm_max_tokens: int = 4096            # javob uchun; kontekstga sig'masa avto-kamayadi
    llm_context_len: int = 20480          # server --max-model-len bilan bir xil bo'lsin
    llm_max_input_chars: int = 55_000     # juda uzun hujjatlar uchun himoya chegarasi (~16k token)
    llm_timeout: float = 600.0

    # [6] Validator
    arithmetic_rel_tolerance: float = 0.01    # 1% — OCR yaxlitlash xatolari uchun
    ocr_match_min_ratio: float = 0.85         # LLM qiymati OCR matnida bormi (fuzzy)

    # [7] Comparison
    compare_fuzzy_threshold: float = 0.85


settings = Settings()

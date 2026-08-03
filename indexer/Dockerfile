# Diqqat: build context loyiha ildizi (docker-compose.yml dagi context: . ni ko'ring)
FROM python:3.11-slim

WORKDIR /app

# HuggingFace keshi - docker-compose'da named volume ulanadi, shunda
# 2.2 GB embedding modeli konteyner qayta yaratilganda qaytadan yuklanmaydi
ENV HF_HOME=/app/hf_cache \
    PYTHONUNBUFFERED=1

# 1. PyTorch CPU versiyasi — embedding (e5-large) EISVO loyihada CPU'da ishlaydi
#    (GPU vllm+surya bilan band). CPU wheel ancha kichik va tez yuklanadi.
RUN pip install --no-cache-dir --default-timeout=2000 --retries 10 \
    torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

# 2. Requirements o'rnatish
COPY indexer/requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 --retries 10 -r requirements.txt

# 3. Kodlarni ko'chirish. scraper/ endi kerak emas: metadata.db mantiqi
#    shared/metadata_db.py ga ko'chirilgan.
COPY indexer/ .
COPY shared/ ./shared/

CMD ["python", "main.py"]

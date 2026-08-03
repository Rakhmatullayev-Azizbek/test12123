# Diqqat: build context loyiha ildizi (docker-compose.yml dagi context: . ni ko'ring)
FROM python:3.11-slim

WORKDIR /app

# HuggingFace keshi - docker-compose'da named volume ulanadi, shunda
# 2.2 GB embedding modeli konteyner qayta yaratilganda qaytadan yuklanmaydi
ENV HF_HOME=/app/hf_cache \
    PYTHONUNBUFFERED=1

# curl - docker-compose healthcheck uchun kerak (python:slim da mavjud emas,
# shu sababli healthcheck avval doim muvaffaqiyatsiz bo'lib konteyner
# "unhealthy" holatida turardi)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU versiyasi (embedding CPU'da) — requirements'dagi torch==2.4.1 shu
# bilan qanoatlantiriladi, katta CUDA wheel yuklanmaydi.
RUN pip install --no-cache-dir --default-timeout=2000 --retries 10 \
    torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

COPY rag_backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_backend/app/ ./app/
COPY shared/ ./shared/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8008"]

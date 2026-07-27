# EISVO OCR pipeline (Surya + pipeline qatlamlari)
# Linux'da PyPI'dagi torch avtomatik CUDA-enabled bo'ladi.
FROM python:3.12-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# web qatlami alohida RUN'da — yuqoridagi katta layer keshi buzilmasligi uchun
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" python-multipart

COPY eisvo_ocr ./eisvo_ocr
COPY main.py .

# Surya modellari shu yerga yuklanadi (compose'da volume qilib beriladi)
ENV HF_HOME=/models
ENV EISVO_VLLM_BASE_URL=http://vllm:8000/v1

ENTRYPOINT ["python", "main.py"]

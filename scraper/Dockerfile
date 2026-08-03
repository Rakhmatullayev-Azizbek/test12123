# Diqqat: build context loyiha ildizi (docker-compose.yml dagi context: . ni ko'ring)
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY scraper/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scraper/ .
COPY shared/ ./shared/

CMD ["python", "main.py"]

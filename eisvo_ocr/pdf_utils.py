"""[1] PDF -> rasmlar. PyMuPDF, ~200 DPI (jadvallar uchun yetarli, CUDA OOM bermaydi)."""
from __future__ import annotations

import io
from pathlib import Path

import pymupdf as fitz
from PIL import Image

from .config import settings


def pdf_to_images(pdf_path: str | Path, dpi: int | None = None) -> list[Image.Image]:
    dpi = dpi or settings.pdf_dpi
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    images: list[Image.Image] = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            images.append(img)
    return images

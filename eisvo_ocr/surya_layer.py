"""[2] SURYA QATLAMI (surya-ocr 0.22.x API) — butun hujjat, batch rejimda:
  - layout detection   -> har sahifada bloklar (LayoutPredictor)
  - text recognition   -> blok HTML + mean-token-prob confidence
                          (RecognitionPredictor, block mode; jadval bloklari
                          tayyor <table> HTML bo'lib keladi)
  - table recognition  -> ustun geometriyasi (TableRecPredictor) — jadval
                          davomini x-koordinata bo'yicha ulash uchun
  - reading order      -> layout .position bo'yicha

Barcha surya chaqiruvlari shu faylda izolyatsiya qilingan — versiya
o'zgarsa faqat shu fayl tuzatiladi.
"""
from __future__ import annotations

from functools import lru_cache

from bs4 import BeautifulSoup
from PIL import Image

from .models import Block, OcrDocument, Page, Table, TableCell

# Surya canonical label -> bizning blok turi
_LABEL_MAP = {
    "Table": "table",
    "Form": "table",
    "Text": "text",
    "ListGroup": "text",
    "Caption": "text",
    "Footnote": "text",
    "Code": "text",
    "Bibliography": "text",
    "TableOfContents": "text",
    "Equation": "text",
    "SectionHeader": "section-header",
    "PageHeader": "page-header",
    "PageFooter": "page-footer",
}
_SKIP_LABELS = {"Picture", "Figure", "Diagram", "ChemicalBlock", "BlankPage"}


@lru_cache(maxsize=1)
def _predictors():
    """Modellarni bir marta yuklaymiz (lazy, GPU'ga)."""
    from surya.layout import LayoutPredictor
    from surya.recognition import RecognitionPredictor
    from surya.table_rec import TableRecPredictor

    return LayoutPredictor(), RecognitionPredictor(), TableRecPredictor()


def run_surya(images: list[Image.Image]) -> OcrDocument:
    layout, recognition, table_rec = _predictors()

    layout_results = layout(images)
    # block mode: har layout blok alohida OCR qilinadi, jadvallar <table> HTML bo'ladi
    ocr_results = recognition(images, layout_results=layout_results)

    doc = OcrDocument()
    table_crops: list[Image.Image] = []
    table_refs: list[Table] = []

    for page_idx, (img, ocr_res) in enumerate(zip(images, ocr_results)):
        page = Page(number=page_idx + 1, width=img.width, height=img.height)
        for blk in sorted(ocr_res.blocks, key=lambda b: b.reading_order):
            if blk.skipped or blk.error or blk.label in _SKIP_LABELS:
                continue
            kind = _LABEL_MAP.get(blk.label, "text")
            bbox = _poly_to_bbox(blk.polygon)
            conf = blk.confidence if blk.confidence is not None else 1.0
            block = Block(kind=kind, bbox=bbox)

            if kind == "table":
                table = _html_to_table(blk.html, bbox, page.number, conf)
                if table.cells:
                    block.table = table
                    x0, y0, x1, y1 = (max(0, int(v)) for v in bbox)
                    table_crops.append(img.crop((x0, y0, x1, y1)))
                    table_refs.append(table)
                else:
                    # jadval parse bo'lmadi — matn sifatida saqlaymiz
                    block.kind = "text"
                    block.lines = _html_to_lines(blk.html, bbox, conf)
            else:
                block.lines = _html_to_lines(blk.html, bbox, conf)

            if block.lines or block.table:
                page.blocks.append(block)
        doc.pages.append(page)

    # ustun geometriyasi: barcha jadval croplari bitta batchda
    if table_crops:
        try:
            tr_results = table_rec(table_crops)
            for table, tr in zip(table_refs, tr_results):
                if tr.error or not tr.cols:
                    continue
                ox = table.bbox[0]
                centers = sorted(
                    (c.bbox[0] + c.bbox[2]) / 2 + ox for c in tr.cols
                )
                table.col_centers_px = centers
        except Exception:  # geometriya ixtiyoriy — merge col-count'ga tayanadi
            pass

    return doc


def _poly_to_bbox(polygon) -> tuple[float, float, float, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return (min(xs), min(ys), max(xs), max(ys))


def _html_to_lines(html: str, bbox, confidence: float):
    """Blok HTML -> OcrLine ro'yxati (har vizual qator alohida)."""
    from .models import OcrLine

    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")
    return [
        OcrLine(line.strip(), bbox, confidence)
        for line in text.splitlines()
        if line.strip()
    ]


def _html_to_table(html: str, bbox, page_number: int, confidence: float) -> Table:
    """Model chiqargan <table> HTML -> Table (rowspan/colspan hisobga olinadi)."""
    table = Table(bbox=bbox, page_number=page_number)
    if not html:
        return table
    soup = BeautifulSoup(html, "html.parser")
    t = soup.find("table") or soup
    occupied: set[tuple[int, int]] = set()  # span'lar band qilgan kataklar
    row_idx = 0
    for tr in t.find_all("tr"):
        col_idx = 0
        for td in tr.find_all(["td", "th"]):
            while (row_idx, col_idx) in occupied:
                col_idx += 1
            rowspan = _span(td.get("rowspan"))
            colspan = _span(td.get("colspan"))
            for r in range(row_idx, row_idx + rowspan):
                for c in range(col_idx, col_idx + colspan):
                    occupied.add((r, c))
            table.cells.append(
                TableCell(
                    row=row_idx,
                    col=col_idx,
                    bbox=bbox,  # HTML'da real hujayra koordinatasi yo'q
                    text=td.get_text(separator=" ").strip(),
                    confidence=confidence,
                    rowspan=rowspan,
                    colspan=colspan,
                    is_header=(td.name == "th"),
                )
            )
            col_idx += colspan
        if tr.find_all(["td", "th"]):
            row_idx += 1
    return table


def _span(value) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1

"""Qatlamlar orasidagi ichki ma'lumot strukturalari (OCR natijalari)."""
from __future__ import annotations

from dataclasses import dataclass, field

BBox = tuple[float, float, float, float]  # x0, y0, x1, y1 (piksel, sahifa koordinatasida)


@dataclass
class OcrLine:
    text: str
    bbox: BBox
    confidence: float


@dataclass
class TableCell:
    row: int
    col: int
    bbox: BBox
    text: str = ""
    confidence: float = 1.0
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False


@dataclass
class Table:
    bbox: BBox
    page_number: int
    cells: list[TableCell] = field(default_factory=list)
    # table_rec bergan ustun markazlari (piksel, sahifa koordinatasida) —
    # HTML'dan parse qilingan hujayralarda real bbox bo'lmaydi
    col_centers_px: list[float] | None = None

    @property
    def n_rows(self) -> int:
        return max((c.row + c.rowspan for c in self.cells), default=0)

    @property
    def n_cols(self) -> int:
        return max((c.col + c.colspan for c in self.cells), default=0)

    def col_x_centers(self, page_width: float) -> list[float]:
        """Har ustunning normalizatsiya qilingan x-markazi (jadval davomini ulash uchun).
        Geometriya noma'lum bo'lsa bo'sh ro'yxat qaytadi — davom-ulash x-tekshiruvni
        o'tkazib yuborib, faqat ustunlar soniga tayanadi."""
        if self.col_centers_px is not None:
            return [x / page_width for x in self.col_centers_px]
        centers: dict[int, list[float]] = {}
        for c in self.cells:
            if c.colspan == 1:
                centers.setdefault(c.col, []).append((c.bbox[0] + c.bbox[2]) / 2 / page_width)
        result = [sum(v) / len(v) for _, v in sorted(centers.items())]
        # HTML'dan kelgan hujayralarda real bbox bo'lmaydi (hammasi jadval bbox'i) —
        # bunday degenerat markazlar bilan solishtirish noto'g'ri rad etishga olib keladi
        if len(result) > 1 and max(result) - min(result) < 0.01:
            return []
        return result

    def rows_as_lists(self) -> list[list[TableCell | None]]:
        grid: list[list[TableCell | None]] = [
            [None] * self.n_cols for _ in range(self.n_rows)
        ]
        for c in self.cells:
            if 0 <= c.row < self.n_rows and 0 <= c.col < self.n_cols:
                grid[c.row][c.col] = c
        return grid


@dataclass
class Block:
    """Layout bloki — reading order bo'yicha tartiblangan."""
    kind: str  # "text" | "table" | "section-header" | "page-header" | "page-footer" | ...
    bbox: BBox
    lines: list[OcrLine] = field(default_factory=list)
    table: Table | None = None

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)


@dataclass
class Page:
    number: int  # 1-based
    width: float
    height: float
    blocks: list[Block] = field(default_factory=list)  # reading order


@dataclass
class OcrDocument:
    pages: list[Page] = field(default_factory=list)

    @property
    def all_lines(self) -> list[OcrLine]:
        out = []
        for p in self.pages:
            for b in p.blocks:
                out.extend(b.lines)
                if b.table:
                    out.extend(
                        OcrLine(c.text, c.bbox, c.confidence) for c in b.table.cells if c.text
                    )
        return out

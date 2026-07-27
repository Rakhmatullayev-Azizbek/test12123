"""[3] DOCUMENT ASSEMBLER — arxitekturaning yuragi.

  - sahifalarni bitta markdown hujjatga birlashtirish
  - jadval davomini ulash: 1-sahifa header sxemasini olib, keyingi
    sahifalardagi headersiz jadvallarni col-count + x-koordinata bo'yicha bog'lash
  - low-confidence so'zlarni belgilash: «so'z⟨0.62⟩»
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import settings
from .models import Block, OcrDocument, Page, Table


@dataclass
class AssembledTable:
    """Birlashtirilgan (davomlari ulangan) jadval."""
    header: list[str]
    rows: list[list[str]]           # marked (low-conf belgili) matn
    pages: list[int] = field(default_factory=list)

    def to_markdown(self) -> str:
        n = max([len(self.header)] + [len(r) for r in self.rows]) if (self.header or self.rows) else 0
        header = (self.header + [""] * n)[:n] or [f"col{i+1}" for i in range(n)]
        lines = [
            "| " + " | ".join(_md_escape(h) for h in header) + " |",
            "|" + "---|" * n,
        ]
        for row in self.rows:
            row = (row + [""] * n)[:n]
            lines.append("| " + " | ".join(_md_escape(c) for c in row) + " |")
        return "\n".join(lines)


@dataclass
class AssembledDocument:
    markdown: str                   # butun hujjat (jadvallar bilan)
    tables: list[AssembledTable]    # faqat jadvallar (LLM Bosqich B uchun)
    plain_text: str                 # belgilarsiz toza matn (OCR-match uchun)

    @property
    def tables_markdown(self) -> str:
        return "\n\n".join(t.to_markdown() for t in self.tables)


def _md_escape(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def mark_low_confidence(text: str, confidence: float) -> str:
    """«so'z⟨0.62⟩» — LLM'ga qaysi so'zlarga ishonmaslikni bildiradi."""
    if text and confidence < settings.low_conf_threshold:
        return f"{text}⟨{confidence:.2f}⟩"
    return text


def assemble(doc: OcrDocument) -> AssembledDocument:
    md_parts: list[str] = []
    plain_parts: list[str] = []
    tables: list[AssembledTable] = []
    # jadval davomini kuzatish: oxirgi jadval + qaysi sahifada tugagan
    open_table: AssembledTable | None = None
    open_table_meta: tuple[int, list[float], int] | None = None  # (page, x_centers, n_cols)
    open_table_md_slot: int | None = None  # markdown'da jadval turadigan joy indeksi

    for page in doc.pages:
        page_blocks = [b for b in page.blocks if b.kind not in ("page-header", "page-footer")]
        for pos, block in enumerate(page_blocks):
            if block.kind == "table" and block.table and block.table.cells:
                table = block.table
                if not _cell_grid(table):  # butunlay bo'sh jadval — shovqin
                    continue
                merged = False
                # davom sharti: jadval sahifadagi BIRINCHI kontent blok bo'lsa
                # va oldingi sahifa jadval bilan tugagan bo'lsa
                if (
                    open_table is not None
                    and open_table_meta is not None
                    and pos == 0
                    and open_table_meta[0] == page.number - 1
                ):
                    merged = _try_merge(open_table, open_table_meta, table, page)
                if merged:
                    open_table.pages.append(page.number)
                    open_table_meta = (page.number, table.col_x_centers(page.width), table.n_cols)
                    md_parts[open_table_md_slot] = open_table.to_markdown()  # type: ignore[index]
                else:
                    at = _table_to_assembled(table, page)
                    tables.append(at)
                    md_parts.append(at.to_markdown())
                    open_table = at
                    open_table_meta = (page.number, table.col_x_centers(page.width), table.n_cols)
                    open_table_md_slot = len(md_parts) - 1
                plain_parts.extend(c.text for c in table.cells if c.text)
                continue

            text = "\n".join(
                mark_low_confidence(l.text, l.confidence) for l in block.lines
            ).strip()
            if not text:
                continue
            if block.kind == "section-header":
                md_parts.append(f"## {text}")
            else:
                md_parts.append(text)
            plain_parts.extend(l.text for l in block.lines)
            # sahifa jadval bilan tugamadi -> davom zanjiri uziladi
            open_table = None
            open_table_meta = None
            open_table_md_slot = None

        # sahifaning OXIRGI kontent bloki jadval bo'lmasa zanjir uzilgan bo'ladi
        # (yuqoridagi tsiklda hal bo'lgan); jadval bilan tugagan bo'lsa saqlanadi

    return AssembledDocument(
        markdown="\n\n".join(md_parts),
        tables=tables,
        plain_text="\n".join(plain_parts),
    )


def _cell_grid(table: Table) -> list[list]:
    """Jadval katagi to'ri, TOZALANGAN: butunlay bo'sh qator va ustunlar olib
    tashlanadi (surya modeli davomiy jadvallarga ba'zan ortiqcha bo'sh
    ustun/qator qo'shib yuboradi — bu davom-ulash col-count tekshiruvini buzadi)."""
    grid = table.rows_as_lists()
    keep_cols = [
        c
        for c in range(table.n_cols)
        if any(row[c] is not None and row[c].text.strip() for row in grid)
    ]
    cleaned = []
    for row in grid:
        cells = [row[c] for c in keep_cols]
        if any(c is not None and c.text.strip() for c in cells):
            cleaned.append(cells)
    return cleaned


def _table_to_assembled(table: Table, page: Page) -> AssembledTable:
    grid = _cell_grid(table)
    str_rows = [
        [mark_low_confidence(c.text, c.confidence) if c else "" for c in row]
        for row in grid
    ]
    if not str_rows:
        return AssembledTable(header=[], rows=[], pages=[page.number])
    # header: table_rec is_header bergan bo'lsa shu, bo'lmasa birinchi qator
    header_row_idx = 0
    for i, row in enumerate(grid):
        if any(c is not None and c.is_header for c in row):
            header_row_idx = i
            break
    header = str_rows[header_row_idx]
    rows = str_rows[:header_row_idx] + str_rows[header_row_idx + 1:]
    return AssembledTable(header=header, rows=rows, pages=[page.number])


def _try_merge(
    open_table: AssembledTable,
    open_meta: tuple[int, list[float], int],
    new_table: Table,
    page: Page,
) -> bool:
    """Davomiy jadvalni oldingisiga ulash: col-count + x-markazlar mosligi."""
    _, prev_centers, _ = open_meta
    grid = _cell_grid(new_table)
    if not grid or len(grid[0]) != len(open_table.header):
        return False
    new_centers = new_table.col_x_centers(page.width)
    if new_centers and prev_centers and len(new_centers) == len(prev_centers):
        # table_rec ustun markazlarini kontentga qarab beradi — sahifalar orasida
        # 0.02-0.04 ga siljishi normal; shuning uchun O'RTACHA farq solishtiriladi
        diffs = [abs(a - b) for a, b in zip(new_centers, prev_centers)]
        if sum(diffs) / len(diffs) > settings.table_merge_col_tolerance:
            return False

    str_rows = [
        [mark_low_confidence(c.text, c.confidence) if c else "" for c in row]
        for row in grid
    ]
    # davomda header takrorlangan bo'lishi mumkin — birinchi qatori
    # mavjud header bilan deyarli bir xil bo'lsa tashlab yuboramiz
    if str_rows and _rows_similar(str_rows[0], open_table.header):
        str_rows = str_rows[1:]
    open_table.rows.extend(str_rows)
    return True


def _rows_similar(a: list[str], b: list[str]) -> bool:
    from rapidfuzz import fuzz

    if not a or not b or len(a) != len(b):
        return False
    joined_a = " ".join(a).lower()
    joined_b = " ".join(b).lower()
    if not joined_a.strip() or not joined_b.strip():
        return False
    return fuzz.ratio(joined_a, joined_b) >= 80

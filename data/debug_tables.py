"""Jadval-ulash diagnostikasi: har jadvalning geometriyasi va tozalangan o'lchami."""
import sys

sys.path.insert(0, "/app")

from eisvo_ocr import pdf_utils
from eisvo_ocr.assembler import _cell_grid
from eisvo_ocr.surya_layer import run_surya

images = pdf_utils.pdf_to_images("/data/test_contract.pdf")
doc = run_surya(images)

for page in doc.pages:
    for i, b in enumerate(page.blocks):
        if b.table:
            t = b.table
            grid = _cell_grid(t)
            print(f"sahifa {page.number} blok {i} (pos in page): kind={b.kind}")
            print(f"  bbox={tuple(round(v) for v in t.bbox)}")
            print(f"  n_rows={t.n_rows} n_cols={t.n_cols} | cleaned: "
                  f"{len(grid)}x{len(grid[0]) if grid else 0}")
            print(f"  col_centers_px={t.col_centers_px}")
            print(f"  col_x_centers(norm)={[round(c, 4) for c in t.col_x_centers(page.width)]}")
    kinds = [b.kind for b in page.blocks]
    print(f"sahifa {page.number} bloklari: {kinds}")

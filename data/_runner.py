"""Batch runner: examples_to_check dagi 1..5 PDF+JSON ni pipeline'dan o'tkazib,
to'liq natijani /data/results/N.json ga yozadi. Har fayldan keyin darhol saqlanadi."""
import json
import sys
import traceback
from pathlib import Path

from eisvo_ocr.pipeline import process_pdf, compare_with_api

BASE = Path("/data/examples_to_check")
OUT = Path("/data/results")
OUT.mkdir(exist_ok=True)


def run_one(n: int) -> None:
    pdf = BASE / f"{n}.pdf"
    api = json.loads((BASE / f"{n}.json").read_text(encoding="utf-8-sig"))
    res = process_pdf(pdf)
    cmp = compare_with_api(res, api)
    out = {
        "file": f"{n}.pdf",
        "extracted": res["extracted"],
        "validation": res["validation"],
        "markdown": res["debug"]["markdown"],
        "comparison": cmp["comparison"],
        "confidence": cmp["confidence"],
    }
    (OUT / f"{n}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    c = cmp["confidence"]
    print(f"[OK] {n}: matched={c['matched']} mismatched={c['mismatched']} "
          f"product_issues={c['product_issues']}", flush=True)


if __name__ == "__main__":
    nums = [int(x) for x in sys.argv[1:]] or [1, 2, 3, 4, 5]
    for n in nums:
        try:
            print(f"=== START {n} ===", flush=True)
            run_one(n)
        except Exception:
            print(f"[FAIL] {n}", flush=True)
            traceback.print_exc()
    print("=== ALL DONE ===", flush=True)

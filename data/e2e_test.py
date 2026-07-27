"""End-to-end test (konteyner ichida): [1] PDF -> [2] Surya -> [3] Assembler -> [4] Deterministik.
Ishga tushirish:
  docker compose run --rm --entrypoint python app /data/e2e_test.py
"""
import json
import logging
import sys
import time

sys.path.insert(0, "/app")
logging.basicConfig(level=logging.INFO, format="%(message)s")

from eisvo_ocr import assembler, deterministic, pdf_utils

t0 = time.time()
images = pdf_utils.pdf_to_images("/data/test_contract.pdf")
print(f"[1] {len(images)} sahifa ({time.time()-t0:.1f}s)")

from eisvo_ocr.surya_layer import run_surya

t0 = time.time()
doc = run_surya(images)
print(f"[2] Surya tugadi ({time.time()-t0:.1f}s)")
for p in doc.pages:
    print(f"    sahifa {p.number}: {[b.kind for b in p.blocks]}")

asm = assembler.assemble(doc)
print(f"[3] {len(asm.tables)} jadval (davomlar ulangandan keyin)")
print("=== MARKDOWN ===")
print(asm.markdown)
print("=== JADVALLAR ===")
print(asm.tables_markdown)

hints = deterministic.extract_candidates(asm.plain_text)
print("[4] Deterministik nomzodlar:")
print(json.dumps(hints, ensure_ascii=False, indent=1))

ok = True
def check(name, cond):
    global ok
    print(f"[{'OK ' if cond else 'FAIL'}] {name}")
    ok = ok and cond

check("INN topildi", "305123456" in hints["inn"])
check("Hisob topildi", "20208000900123456789" in hints["accounts"])
check("SWIFT x2", {"KACHUZ22", "DEUTDEFF"} <= set(hints["swift"]))
check("Sana", "2024-03-25" in hints["dates"])
check("DAT->DPU", "DPU" in hints["incoterms"])
check("Jadval davomi ulandi (1 jadval)", len(asm.tables) == 1)
if asm.tables:
    check("5 ma'lumot qatori (2+3)", len(asm.tables[0].rows) == 5)
sys.exit(0 if ok else 1)

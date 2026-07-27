"""CLI kirish nuqtasi.

Foydalanish:
  python main.py contract.pdf                      # faqat extraction + validation
  python main.py contract.pdf --api api.json       # API ma'lumoti bilan taqqoslash
  python main.py contract.pdf -o result.json --debug
"""
import argparse
import json
import logging
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="EISVO shartnoma PDF extraction")
    parser.add_argument("pdf", type=Path, help="Shartnoma PDF fayli")
    parser.add_argument("--api", type=Path, help="API javobi (JSON fayl) — taqqoslash uchun")
    parser.add_argument("-o", "--output", type=Path, help="Natijani JSON faylga yozish")
    parser.add_argument("--md", type=Path, help="OCR markdown'ini alohida faylga saqlash")
    parser.add_argument("--debug", action="store_true", help="Markdown/hints ham chiqsin")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.pdf.exists():
        print(f"Fayl topilmadi: {args.pdf}", file=sys.stderr)
        return 1

    from eisvo_ocr import process_pdf
    from eisvo_ocr.pipeline import compare_with_api

    result = process_pdf(args.pdf)
    if args.md:
        args.md.write_text(result["debug"]["markdown"], encoding="utf-8")
        print(f"Markdown yozildi: {args.md}")

    if args.api:
        api_data = json.loads(args.api.read_text(encoding="utf-8"))
        result = compare_with_api(result, api_data)
    elif not args.debug:
        result.pop("debug", None)

    out = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.write_text(out, encoding="utf-8")
        print(f"Natija yozildi: {args.output}")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

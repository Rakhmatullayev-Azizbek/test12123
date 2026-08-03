"""
ILOVA (appendix) ichidagi <table>larni RAG uchun foydali formatga aylantiradi.

Muammo: xom jadval qatori ("AVSTRALIYA | AQSH dollari | 45,00 | 20,00 | 100,00")
embedding uchun ma'nosiz - foydalanuvchi savoli ("AQShga xizmat safari uchun
qancha to'lanadi?") bilan semantik mos kelmaydi.

Yechim: har bir qatorni sarlavha (birinchi qator) bilan birlashtirib, tabiiy
tilga o'xshash jumlaga aylantiramiz.

lex.uz jadvallari xususiyatlari (amaliy tekshirilgan):
    - <th> ishlatilmaydi, sarlavha ham <td> bilan beriladi
    - colspan tufayli sarlavha ustunlari soni qatordagi hujayralardan kam bo'lishi mumkin
    - nol kenglikdagi belgilar (U+200E/U+200F) va bo'sh hujayralar ko'p uchraydi
    - ba'zi "jadval"lar bitta qatorli - bu tahrir kiritish iqtibosi, mazmun emas
"""
import re

from bs4 import BeautifulSoup

# "T/r" (tartib raqami) kabi sof indeks ustunlar mazmunga hech narsa qo'shmaydi
IGNORED_HEADERS = {"t/r", "т/р", "№", "no", "n", "-", "—", ""}

# Nol kenglikdagi va boshqa ko'rinmas belgilar
_INVISIBLE = dict.fromkeys(map(ord, "‎‏​­"), None)

# Sof tartib raqami hujayrasi: "1", "1.", "12)" - mazmun tashimaydi
_ORDINAL_CELL = re.compile(r"^\d+\s*[.)]?$")


def _cell_text(cell) -> str:
    text = cell.get_text(separator=" ", strip=True)
    text = text.translate(_INVISIBLE).replace("\xa0", " ")
    # Blank-forma to'ldirish chiziqlari ("________ 20__ yil") mazmun tashimaydi
    has_blank_line = bool(re.search(r"_{3,}", text))
    text = re.sub(r"_{3,}", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    # Diqqat: bu tekshiruv faqat to'ldirish chizig'i bo'lgan hujayralarga
    # qo'llanadi. Aks holda tartib raqami ("1.") ham o'chib ketardi va
    # sarlavha/ma'lumot chegarasini aniqlash buzilardi.
    if has_blank_line and len(re.findall(r"[^\W_]", text)) < 3:
        return ""
    # Mazmunsiz hujayralar: faqat tinish belgilari (“ yoki ”;) yoki blank-forma
    # to'ldirish chiziqlari ("______"). Diqqat: \w pastki chiziqni ham qamraydi,
    # shuning uchun [^\W_] - "harf yoki raqam, lekin pastki chiziq emas" ishlatiladi.
    if text and not re.search(r"[^\W_]", text):
        return ""
    return text


def _row_cells(row) -> list[str]:
    return [_cell_text(c) for c in row.find_all(["td", "th"])]


# Sarlavha nomi shundan uzun bo'lsa qisqartiriladi - aks holda har bir qator
# jumlasi sarlavha matni bilan to'lib ketadi
MAX_HEADER_CHARS = 120


def _expand_grid(rows) -> list[list[str]]:
    """
    Jadvalni colspan/rowspan hisobga olib to'g'ri to'rtburchak to'rga yoyadi.

    Bu shart, chunki lex.uz jadvallarida ikki qatorli sarlavha ishlatiladi:

        qator 0: T/r(rowspan=2) | Mamlakat(rowspan=2) | Sutkalik xarajat(colspan=2) | Turar joy(rowspan=2)
        qator 1:                                         tashqarida | ichkarida
        data:    1. | AVSTRALIYA | 45,00 | 20,00 | 100,00

    Yoyilmasa sarlavhalar ustunlarga surilib ketadi va qiymatlar noto'g'ri
    nom bilan belgilanadi (masalan turar joy me'yori "ustun 6" bo'lib qoladi,
    natijada LLM javobda noto'g'ri raqamni keltiradi).
    """
    grid: dict[tuple[int, int], str] = {}
    occupied: set[tuple[int, int]] = set()

    for r, row in enumerate(rows):
        c = 0
        for cell in row.find_all(["td", "th"]):
            while (r, c) in occupied:
                c += 1
            text = _cell_text(cell)
            try:
                colspan = max(1, int(cell.get("colspan") or 1))
                rowspan = max(1, int(cell.get("rowspan") or 1))
            except (TypeError, ValueError):
                colspan = rowspan = 1
            # Buzilgan HTML cheksiz to'r yasamasligi uchun cheklaymiz
            colspan, rowspan = min(colspan, 50), min(rowspan, 50)

            for dr in range(rowspan):
                for dc in range(colspan):
                    occupied.add((r + dr, c + dc))
                    grid[(r + dr, c + dc)] = text
            c += colspan

    if not grid:
        return []

    width = max(c for _, c in grid) + 1
    return [
        [grid.get((r, c), "") for c in range(width)] for r in range(len(rows))
    ]


def _compose_headers(header_rows: list[list[str]], width: int) -> list[str]:
    """
    Bir necha sarlavha qatorini ustun bo'yicha birlashtiradi.

    rowspan tufayli takrorlangan matn bir marta olinadi, colspan guruhining
    umumiy nomi bilan quyi sarlavha " — " orqali qo'shiladi.
    """
    headers: list[str] = []
    for c in range(width):
        parts: list[str] = []
        for row in header_rows:
            value = row[c].strip() if c < len(row) else ""
            if value and value not in parts:
                parts.append(value)
        name = " — ".join(parts)
        if len(name) > MAX_HEADER_CHARS:
            name = name[:MAX_HEADER_CHARS].rsplit(" ", 1)[0].rstrip(" ,;—-")
        headers.append(name)
    return headers


def _looks_like_data_row(cells: list[str]) -> bool:
    """
    Birinchi qator sarlavha emas, oddiy ma'lumot ekanini aniqlaydi.

    lex.uz'da ba'zi "jadval"lar aslida raqamlangan ro'yxat: qator 0 dan
    boshlab "1. | Avstraliya Ittifoqi" ko'rinishida keladi va sarlavha
    qatori umuman yo'q. Bunda birinchi qatorni sarlavha deb olish
    ma'lumotning bir qatorini yo'qotadi va barcha ustun nomlarini buzadi.
    """
    filled = [c for c in cells if c]
    if not filled:
        return False
    return bool(_ORDINAL_CELL.match(filled[0]))


def _normalize_headers(headers: list[str], width: int) -> list[str]:
    """
    Sarlavha ro'yxatini qator kengligiga moslashtiradi va takrorlarni farqlaydi.
    colspan tufayli sarlavha qisqa bo'lsa - "ustun N" bilan to'ldiriladi.
    """
    result: list[str] = []
    seen: dict[str, int] = {}
    for i in range(width):
        name = headers[i].strip() if i < len(headers) else ""
        if not name:
            name = f"ustun {i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 1
        result.append(name)
    return result


def table_to_sentences(table_html: str, table_title: str, document_id: str) -> list[dict]:
    """
    Bitta <table>ni qator-qator tabiiy jumlalarga aylantiradi.

    Qaytaradi: [{"document_id": ..., "section_type": "jadval",
                 "embed_text": "...", "context_text": "...", "row_data": {...}}, ...]
    """
    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table")
    if table is None:
        return []

    rows = table.find_all("tr")
    # Bitta qatorli "jadval" - tahrir iqtibosi, mazmunli ma'lumot emas
    if len(rows) < 2:
        return []

    grid = _expand_grid(rows)
    if not grid:
        return []
    width = len(grid[0])

    # Ma'lumot qaysi qatordan boshlanishini aniqlaymiz: birinchi tartib
    # raqami bilan boshlanadigan qator. Undan oldingi hamma narsa - sarlavha.
    data_start = next(
        (i for i, cells in enumerate(grid) if _looks_like_data_row(cells)), None
    )

    if data_start is None:
        # Tartib raqamli qator yo'q (masalan blank-forma jadvallari) -
        # birinchi qatorni sarlavha deb olamiz
        data_start = 1
        if not any(grid[0]):
            if len(grid) < 3:
                return []
            data_start = 2

    headerless = data_start == 0
    header_cells = (
        [] if headerless else _compose_headers(grid[:data_start], width)
    )

    title = (table_title or "").strip()
    prefix = f"{title} — " if title else ""

    results = []
    for cells in grid[data_start:]:
        if not cells or not any(cells):
            continue

        if headerless:
            # Tartib raqamlarini tashlab, qolgan qiymatlarni birlashtiramiz
            values = [c for c in cells if c and not _ORDINAL_CELL.match(c)]
            if not values:
                continue
            sentence = f"{prefix}" + "; ".join(values) + "."
            row_data = {f"ustun {i + 1}": v for i, v in enumerate(cells) if v}
        else:
            headers = _normalize_headers(header_cells, len(cells))
            row_data = {h: v for h, v in zip(headers, cells) if v}

            meaningful_pairs = [
                (header, value)
                for header, value in zip(headers, cells)
                if value and header.strip().lower() not in IGNORED_HEADERS
            ]
            if not meaningful_pairs:
                continue

            # Birinchi mazmunli ustun - "nom" (mamlakat/toifa), qolganlari - qiymatlar
            _, name_value = meaningful_pairs[0]
            value_parts = [f"{header}: {value}" for header, value in meaningful_pairs[1:]]

            # Faqat nom bo'lib, hech qanday qiymat bo'lmasa - jumla ma'nosiz
            # ("... — (bank nomi). ." ko'rinishidagi axlatning oldini oladi)
            if not value_parts:
                continue

            sentence = f"{prefix}{name_value}. " + "; ".join(value_parts) + "."

        results.append({
            "document_id": document_id,
            "section_type": "jadval",
            "embed_text": sentence,
            "context_text": sentence,  # jadval qatorlari uchun bob-context tegishli emas
            "row_data": row_data,
        })

    return results

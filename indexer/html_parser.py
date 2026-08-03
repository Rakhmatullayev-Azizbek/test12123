"""
lex.uz hujjat sahifasini (https://lex.uz/docs/{id}) parslash.

Ikki xil hujjat strukturasini qo'llab-quvvatlaydi:

1) Kodeks/Qonun turi - har bir modda CLAUSE_DEFAULT tegi bilan aniq belgilangan
   (masalan Saylov kodeksi: <div class="CLAUSE_DEFAULT">1-modda...</div>)

2) Buyruq/Qaror/Nizom turi - CLAUSE_DEFAULT yo'q, band raqami ACT_TEXT matnining
   boshida oddiy raqam sifatida keladi (masalan: "30. Tashkilotlarning rahbarlari...")

MUHIM - lx_elem2 muammosi:
    lex.uz har bir mazmunli div ichiga yashirin UI panelini joylashtiradi:

        <div class="ACT_TEXT lx_elem">
          <div class="lx_elem2">          <- UI: "Ҳужжатга таклиф юбориш" va h.k.
            <div class="lx_elem3">...</div>
          </div>
          <div id="-5451303">1. Haqiqiy matn...</div>
        </div>

    Agar bu panel olib tashlanmasa, get_text() natijasi kirill UI matni bilan
    boshlanadi va band raqamini aniqlaydigan regexp hech qachon mos kelmaydi -
    natijada butun hujjat bitta ulkan chunk'ga aylanadi. _clean_text() shu
    panelni majburiy tozalaydi.

    Ichki <div id="..."> ning id'si - lex.uz element identifikatori. Uni
    saqlaymiz va aniq bandga havola qilish uchun ishlatamiz:
    https://lex.uz/docs/-5445273#-5451303

Jadvallar (ILOVA) BY_DEFAULT / TABLE_STD / TABLE_STD2 divlari ichida keladi va
table_transformer.py orqali qator-qator tabiiy jumlalarga aylantiriladi.

Natija (to_chunks): har bir chunk uchun
    - embed_text:   qidiruv/embedding uchun matn (MAX_CHUNK_CHARS gacha)
    - context_text: LLM'ga beriladigan kengaytirilgan kontekst (qo'shni bandlar)
"""
import copy
import re
import uuid
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from shared.chunking import split_text
from shared.config import settings
from table_transformer import table_to_sentences

# Qdrant point ID'lari uchun barqaror namespace (deterministik uuid5)
CHUNK_ID_NAMESPACE = uuid.UUID("5f6d7a1c-0b2e-4f3a-9c4d-8e1a2b3c4d5e")

# Band raqamini ACT_TEXT boshida aniqlaydigan pattern.
# Mos keladi: "1. ", "30. ", "8¹. " (yuqori indeksli raqamlar bilan ham).
# <sup> teglari _clean_text ichida Unicode yuqori indeksga aylantirilgani uchun
# bu yerda HTML tegini kutish shart emas.
_SUP_CHARS = "¹²³⁰⁴⁵⁶⁷⁸⁹"
BAND_NUMBER_PATTERN = re.compile(rf"^(\d+[{_SUP_CHARS}]*)\.\s")

# "1-modda", "8¹-modda" ko'rinishidagi modda sarlavhasi
MODDA_NUMBER_PATTERN = re.compile(rf"^(\d+[{_SUP_CHARS}]*)\s*-\s*modda")

# <sup>1</sup> -> "¹" almashtirish jadvali
_SUPERSCRIPT_MAP = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
}

# lex.uz UI paneli va tanlanmaydigan xizmat elementlari - matndan chiqarib tashlanadi
JUNK_SELECTORS = ".lx_elem2, .lx_no_select, script, style"

# Chiqarib tashlanadigan (matn mazmuniga aloqasi yo'q) CSS klasslar
IGNORED_CLASSES = {
    "INDEXES_ON_REF",   # klassifikator kodlari (OKOZ/TSZ), display:none
    "COMMENT",          # "Oldingi tahrirga qarang" havolalari
    "CHANGES_ORIGINS",  # tahrir tarixi
}

# Jadval saqlashi mumkin bo'lgan konteyner klasslar. Bular o'z-o'zidan matn
# sifatida qiziq emas, lekin ichida <table> bo'lsa - qayta ishlanadi.
TABLE_CONTAINER_CLASSES = {"BY_DEFAULT", "TABLE_STD", "TABLE_STD2"}

# Hujjat/ilova sarlavhasini beradigan klasslar
TITLE_CLASSES = {"ACT_TITLE", "ACT_TITLE_APPL", "APPL_BANNER_LANDSCAPE_TITLE"}


@dataclass
class Band:
    """Bitta modda/band - eng kichik chunklash birligi."""
    document_id: str
    document_title: str
    section_type: str  # "buyruq" | "nizom" | "kodeks"
    chapter: str | None
    chapter_index: int          # bob nomi takrorlansa ham to'qnashmasligi uchun
    band_number: str | None
    band_title: str | None      # CLAUSE_DEFAULT bo'lsa uning matni
    element_id: str | None = None  # lex.uz ichki element id (havola anchor'i)
    embed_text: str = ""
    paragraphs: list[str] = field(default_factory=list)

    def finalize_embed_text(self) -> None:
        parts = []
        if self.band_title:
            parts.append(self.band_title)
        parts.extend(self.paragraphs)
        self.embed_text = "\n".join(p.strip() for p in parts if p.strip())


@dataclass
class TableRow:
    """Jadvalning bitta qatoridan olingan tabiiy jumla."""
    document_id: str
    document_title: str
    table_title: str
    element_id: str | None
    embed_text: str
    row_data: dict


@dataclass
class ParsedDocument:
    document_id: str
    document_title: str
    accepting_body: str | None
    act_form: str | None
    bands: list[Band] = field(default_factory=list)
    table_rows: list[TableRow] = field(default_factory=list)

    # --- kontekst qurish ---

    def _build_context(self, index: int) -> str:
        """
        Band atrofidagi kontekstni yig'adi: bandning o'zi + shu bobdagi
        qo'shni bandlar, MAX_CONTEXT_CHARS chegarasigacha ikki tomonga
        navbatma-navbat kengayadi.

        Butun bobni tashlash (eski yondashuv) kontekstni 15-30 ming belgigacha
        o'stirib yuborardi; bu yerda chegara boshqariladi.
        """
        band = self.bands[index]
        budget = settings.MAX_CONTEXT_CHARS

        pieces: list[str] = [band.embed_text]
        used = len(band.embed_text)

        left, right = index - 1, index + 1
        while used < budget and (left >= 0 or right < len(self.bands)):
            progressed = False

            if left >= 0:
                nb = self.bands[left]
                if nb.chapter_index == band.chapter_index and used + len(nb.embed_text) <= budget:
                    pieces.insert(0, nb.embed_text)
                    used += len(nb.embed_text)
                    progressed = True
                left -= 1

            if right < len(self.bands) and used < budget:
                nb = self.bands[right]
                if nb.chapter_index == band.chapter_index and used + len(nb.embed_text) <= budget:
                    pieces.append(nb.embed_text)
                    used += len(nb.embed_text)
                    progressed = True
                right += 1

            if not progressed and left < 0 and right >= len(self.bands):
                break

        header = []
        if self.document_title:
            header.append(self.document_title)
        if band.chapter:
            header.append(band.chapter)

        body = "\n\n".join(p for p in pieces if p)
        return ("\n".join(header) + "\n\n" + body).strip() if header else body

    def to_chunks(self) -> list[dict]:
        """
        Indekslashga tayyor chunk ro'yxati.

        Har bir band embed_text bo'yicha MAX_CHUNK_CHARS gacha bo'laklanadi
        (e5-large 512 token chegarasi), lekin context_text kengaytirilgan
        bo'lib qoladi - hybrid chunking yondashuvi.
        """
        chunks: list[dict] = []

        for idx, band in enumerate(self.bands):
            context_text = self._build_context(idx)
            parts = split_text(band.embed_text)
            for part_no, part in enumerate(parts):
                chunks.append(
                    _make_chunk(
                        document_id=band.document_id,
                        document_title=band.document_title,
                        section_type=band.section_type,
                        kind="band",
                        chapter=band.chapter,
                        band_number=band.band_number,
                        element_id=band.element_id,
                        embed_text=part,
                        context_text=context_text,
                        part_no=part_no,
                        part_total=len(parts),
                        local_key=f"band:{idx}:{part_no}",
                    )
                )

        for idx, row in enumerate(self.table_rows):
            for part_no, part in enumerate(split_text(row.embed_text)):
                chunks.append(
                    _make_chunk(
                        document_id=row.document_id,
                        document_title=row.document_title,
                        section_type="jadval",
                        kind="jadval",
                        chapter=row.table_title,
                        band_number=None,
                        element_id=row.element_id,
                        embed_text=part,
                        context_text=part,
                        part_no=part_no,
                        part_total=1,
                        local_key=f"table:{idx}:{part_no}",
                        row_data=row.row_data,
                    )
                )

        return chunks


def _make_chunk(*, document_id: str, local_key: str, **payload) -> dict:
    """Deterministik chunk_id bilan chunk dict yasaydi (qayta indekslashda ustiga yozish uchun)."""
    chunk_id = str(uuid.uuid5(CHUNK_ID_NAMESPACE, f"{document_id}|{local_key}"))
    anchor = payload.get("element_id")
    source_url = f"{settings.LEX_UZ_BASE_URL}/docs/{document_id}"
    if anchor:
        source_url = f"{source_url}#{anchor}"
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "source_url": source_url,
        **payload,
    }


# ---------- HTML tozalash ----------


def _get_div_class(tag: Tag) -> str | None:
    classes = tag.get("class") or []
    return classes[0] if classes else None


def _strip_junk(tag: Tag) -> Tag:
    """lx_elem2 (UI paneli) va boshqa xizmat elementlarini olib tashlangan nusxa qaytaradi."""
    clone = copy.copy(tag)
    for junk in clone.select(JUNK_SELECTORS):
        junk.decompose()
    return clone


def _normalize_superscripts(tag: Tag) -> None:
    """<sup>1</sup> -> "¹" - get_text() da "8 1 -modda" bo'lib buzilmasligi uchun."""
    for sup in tag.find_all("sup"):
        raw = sup.get_text(strip=True)
        if raw and all(ch in _SUPERSCRIPT_MAP for ch in raw):
            sup.replace_with("".join(_SUPERSCRIPT_MAP[ch] for ch in raw))


def _clean_text(tag: Tag) -> str:
    """
    Mazmunli matnni ajratib oladi: UI panelini olib tashlaydi, yuqori
    indekslarni normallashtiradi, ortiqcha bo'shliqni siqadi.
    """
    clone = _strip_junk(tag)
    _normalize_superscripts(clone)
    text = clone.get_text(separator=" ", strip=True)
    # Ko'p bo'shliq/nozik bo'shliqlarni bitta oddiy bo'shliqqa keltiramiz
    text = text.replace("‎", "").replace("‏", "").replace("\xa0", " ")
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _find_element_id(tag: Tag) -> str | None:
    """
    Mazmun saqlovchi ichki <div id="-5451303"> ning id'sini qaytaradi.
    Bu lex.uz element anchor'i - aniq bandga havola qilish uchun.
    """
    for inner in tag.find_all("div", id=True):
        candidate = inner.get("id", "")
        if re.fullmatch(r"-?\d+", candidate):
            return candidate
    return None


TABLE_TITLE_MAX_CHARS = 180


def _table_title(appendix_title: str, document_title: str, preceding_text: str) -> str:
    """
    Jadval uchun qisqa va mazmunli sarlavha tanlaydi.

    Ustuvorlik: ilova sarlavhasi -> hujjat sarlavhasi. Jadval oldidagi matn
    faqat u haqiqatan sarlavhaga o'xshasa (qisqa) qo'shiladi - aks holda butun
    band matni jadvalning har bir qatoriga ko'chib, embedding'ni buzadi.
    """
    base = (appendix_title or document_title or "").strip()

    lead = (preceding_text or "").strip()
    # Uzun xatboshi sarlavha emas; nuqta bilan tugagan to'liq jumla ham emas
    if 20 <= len(lead) <= 120 and not lead.endswith("."):
        base = f"{base} — {lead}" if base else lead

    base = base.strip(" —")
    if len(base) > TABLE_TITLE_MAX_CHARS:
        # So'zning o'rtasidan kesmaslik uchun oxirgi bo'shliqqa qadar qisqartiramiz
        cut = base[:TABLE_TITLE_MAX_CHARS].rsplit(" ", 1)[0]
        base = (cut or base[:TABLE_TITLE_MAX_CHARS]).rstrip(" ,;—-")
    return base


def _extract_band_number(text: str) -> str | None:
    match = MODDA_NUMBER_PATTERN.match(text) or BAND_NUMBER_PATTERN.match(text)
    return match.group(1) if match else None


# ---------- Asosiy parser ----------


def parse_lex_document(html: str, document_id: str) -> ParsedDocument:
    """
    Asosiy kirish nuqtasi. HTML matnini oladi, ParsedDocument qaytaradi.

    document_id: masalan "-2818868" (URL dagi /docs/{id} qismi)
    """
    soup = BeautifulSoup(html, "html.parser")
    div_cont = soup.find("div", id="divCont")
    if div_cont is None:
        raise ValueError(f"divCont topilmadi - hujjat {document_id} kutilgan formatda emas")

    document_title = ""
    accepting_body: str | None = None
    act_form: str | None = None
    section_type = "kodeks"  # ACCEPTING_BODY uchrasa "buyruq"ga o'zgaradi

    current_chapter: str | None = None
    chapter_index = 0
    current_band: Band | None = None
    bands: list[Band] = []
    table_rows: list[TableRow] = []

    # Jadvalga sarlavha berish uchun oxirgi ko'rilgan mazmunli sarlavha/matn
    last_appendix_title: str = ""
    last_text_line: str = ""

    def flush_current_band() -> None:
        nonlocal current_band
        if current_band is not None:
            current_band.finalize_embed_text()
            if current_band.embed_text:
                bands.append(current_band)
        current_band = None

    def new_band(number: str | None, title: str | None, element_id: str | None) -> Band:
        return Band(
            document_id=document_id,
            document_title=document_title,
            section_type=section_type,
            chapter=current_chapter,
            chapter_index=chapter_index,
            band_number=number,
            band_title=title,
            element_id=element_id,
        )

    for child in div_cont.find_all("div", recursive=False):
        css_class = _get_div_class(child)
        if css_class is None:
            continue

        # --- Jadvallar: BY_DEFAULT/TABLE_STD ichida yashiringan bo'lishi mumkin ---
        tables = child.find_all("table")
        if tables:
            flush_current_band()
            for table in tables:
                title = _table_title(last_appendix_title, document_title, last_text_line)
                for row in table_to_sentences(str(table), title, document_id):
                    table_rows.append(
                        TableRow(
                            document_id=document_id,
                            document_title=document_title,
                            table_title=title,
                            element_id=_find_element_id(child),
                            embed_text=row["embed_text"],
                            row_data=row["row_data"],
                        )
                    )
            continue

        if css_class in IGNORED_CLASSES or css_class in TABLE_CONTAINER_CLASSES:
            continue

        if css_class == "ACCEPTING_BODY":
            accepting_body = _clean_text(child)
            section_type = "buyruq"
            continue

        if css_class == "ACT_FORM":
            text = _clean_text(child)
            if text and not act_form:
                act_form = text
            if text and "nizom" in text.lower():
                section_type = "nizom"
            continue

        if css_class in TITLE_CLASSES:
            text = _clean_text(child)
            if not text:
                continue
            if not document_title:
                document_title = text
            else:
                # Ilova sarlavhasi - keyingi jadvallarga nom berish uchun eslab qolamiz
                last_appendix_title = text
            continue

        if css_class == "TEXT_HEADER_DEFAULT":
            # Yangi bob boshlandi -> joriy bandni yopamiz
            flush_current_band()
            current_chapter = _clean_text(child)
            chapter_index += 1
            continue

        if css_class == "CLAUSE_DEFAULT":
            # Yangi modda (Kodeks/Qonun turi)
            flush_current_band()
            band_title = _clean_text(child)
            if not band_title:
                continue
            current_band = new_band(
                _extract_band_number(band_title), band_title, _find_element_id(child)
            )
            last_text_line = band_title
            continue

        if css_class == "ACT_TEXT":
            text = _clean_text(child)
            if not text:
                continue
            last_text_line = text

            band_number = _extract_band_number(text)
            if band_number is not None:
                # Buyruq/Nizom turi: raqam matn boshida -> yangi band
                flush_current_band()
                current_band = new_band(band_number, None, _find_element_id(child))
                current_band.paragraphs.append(text)
            else:
                # Davomi paragraf - joriy bandga qo'shiladi.
                # Hali band boshlanmagan bo'lsa (muqaddima/kirish) - yaratamiz.
                if current_band is None:
                    current_band = new_band(None, None, _find_element_id(child))
                current_band.paragraphs.append(text)
            continue

        # Qolgan klasslar (FOOTNOTE, ACT_ESSENTIAL_ELEMENTS, EXTRACT va h.k.) -
        # matn sifatida joriy bandga qo'shiladi, agar mazmuni bo'lsa.
        text = _clean_text(child)
        if text:
            if current_band is None:
                current_band = new_band(None, None, _find_element_id(child))
            current_band.paragraphs.append(text)

    flush_current_band()

    # document_title parse oxirida aniqlangan bo'lishi mumkin - bandlarga tarqatamiz
    if document_title:
        for band in bands:
            if not band.document_title:
                band.document_title = document_title
        for row in table_rows:
            if not row.document_title:
                row.document_title = document_title

    return ParsedDocument(
        document_id=document_id,
        document_title=document_title,
        accepting_body=accepting_body,
        act_form=act_form,
        bands=bands,
        table_rows=table_rows,
    )

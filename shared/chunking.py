"""
Uzun matnlarni embedding modeli chegarasiga sig'adigan bo'laklarga ajratish.

multilingual-e5-large 512 token qabul qiladi (~1200-1500 belgi o'zbek matni
uchun). Undan uzun matn model ichida jimgina qirqiladi va chunk'ning qolgan
qismi hech qachon indekslanmaydi. Shuning uchun bo'laklashni o'zimiz bajaramiz.

Bo'laklash gap/xatboshi chegaralarini hurmat qiladi, ya'ni jumlaning
o'rtasidan kesmaydi (imkoni bo'lsa).
"""
import re

from .config import settings

# Xatboshi va jumla oxiri chegaralari (o'zbek matni uchun)
_PARAGRAPH_SPLIT = re.compile(r"\n+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+")


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Oxirgi chora: hech qanday chegara topilmasa, belgi bo'yicha kesish."""
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _split_units(text: str, max_chars: int) -> list[str]:
    """
    Matnni max_chars dan oshmaydigan eng katta mazmunli birliklarga ajratadi:
    xatboshi -> jumla -> (oxirgi chora) belgi.
    """
    units: list[str] = []
    for para in _PARAGRAPH_SPLIT.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
            continue
        for sentence in _SENTENCE_SPLIT.split(para):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= max_chars:
                units.append(sentence)
            else:
                units.extend(_hard_split(sentence, max_chars))
    return units


def split_text(
    text: str,
    max_chars: int | None = None,
    overlap_chars: int | None = None,
) -> list[str]:
    """
    Matnni max_chars dan oshmaydigan bo'laklarga ajratadi.

    Ketma-ket bo'laklar orasida overlap_chars belgilik ustma-ustlik qoldiriladi -
    chegarada turgan ma'lumot ikkala bo'lakda ham uchraydi, shunda qidiruv
    uni topib qolish ehtimoli oshadi.

    Matn allaqachon qisqa bo'lsa - [text] qaytadi (bo'linmaydi).
    """
    max_chars = max_chars or settings.MAX_CHUNK_CHARS
    overlap_chars = settings.CHUNK_OVERLAP_CHARS if overlap_chars is None else overlap_chars

    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    units = _split_units(text, max_chars)
    if not units:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for unit in units:
        # +1 - birlashtirishdagi bo'shliq/qator uchun
        if current and current_len + len(unit) + 1 > max_chars:
            chunks.append("\n".join(current))
            # Ustma-ustlik: oxirgi birliklardan overlap_chars gacha olib qolamiz
            carry: list[str] = []
            carry_len = 0
            for prev in reversed(current):
                if carry_len + len(prev) > overlap_chars:
                    break
                carry.insert(0, prev)
                carry_len += len(prev) + 1
            current = carry
            current_len = carry_len

        current.append(unit)
        current_len += len(unit) + 1

    if current:
        tail = "\n".join(current)
        # Ustma-ustlik tufayli oxirgi bo'lak butunlay takror bo'lib qolmasin
        if not chunks or tail not in chunks[-1]:
            chunks.append(tail)

    return chunks

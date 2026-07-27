"""EISVO/bojxona klassifikator kodlari <-> PDF'dagi matn qiymatlari tarjimasi.

API kodlarda qaytaradi (currCode "840", countryCode "156", incotermsCode "08",
unitCode "796") — PDF'da esa "USD", "Китай", "CIP", "шт" deb yoziladi.
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz

# кирилл -> лотин: skript-agnostik solishtirish uchun (compare.py ham ishlatadi).
# Ikkala tomon bir xil qoidada o'girilgani uchun ru/uz tafovutlari fuzzy'da yutiladi;
# o'lchov birliklarida esa aynan mos keladi (кг->kg, шт->sht, м2->m2).
_CYR2LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "ғ": "g", "д": "d", "е": "e",
    "ё": "yo", "ж": "j", "з": "z", "и": "i", "й": "y", "к": "k", "қ": "q",
    "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s",
    "т": "t", "у": "u", "ў": "o", "ф": "f", "х": "x", "ҳ": "h", "ц": "ts",
    "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "i", "ь": "", "э": "e",
    "ю": "yu", "я": "ya",
}


def _translit(s: str) -> str:
    return "".join(_CYR2LAT.get(ch, ch) for ch in s)

# --- ISO 4217 raqamli -> harfli valyuta ---
CURRENCY_NUM2ALPHA = {
    "840": "USD", "978": "EUR", "860": "UZS", "643": "RUB", "156": "CNY",
    "392": "JPY", "826": "GBP", "756": "CHF", "398": "KZT", "417": "KGS",
    "972": "TJS", "934": "TMT", "944": "AZN", "051": "AMD", "981": "GEL",
    "949": "TRY", "784": "AED", "410": "KRW", "356": "INR", "124": "CAD",
    "036": "AUD", "702": "SGD", "344": "HKD", "985": "PLN", "203": "CZK",
}

# --- ISO 3166 raqamli davlat kodi -> nomlar (ru/uz/en variantlari) ---
COUNTRY_NUM2NAMES: dict[str, list[str]] = {
    "156": ["CN", "China", "Китай", "Xitoy", "КНР", "PRC"],
    "860": ["UZ", "Uzbekistan", "Узбекистан", "O'zbekiston", "Ўзбекистон"],
    "643": ["RU", "Russia", "Россия", "Rossiya", "Российская Федерация", "РФ"],
    "398": ["KZ", "Kazakhstan", "Казахстан", "Qozog'iston"],
    "417": ["KG", "Kyrgyzstan", "Кыргызстан", "Qirg'iziston", "Киргизия"],
    "762": ["TJ", "Tajikistan", "Таджикистан", "Tojikiston"],
    "795": ["TM", "Turkmenistan", "Туркменистан", "Turkmaniston"],
    "004": ["AF", "Afghanistan", "Афганистан", "Afg'oniston"],
    "276": ["DE", "Germany", "Германия", "Germaniya", "BRD", "ФРГ"],
    "792": ["TR", "Turkey", "Türkiye", "Турция", "Turkiya"],
    "410": ["KR", "Korea", "Южная Корея", "Koreya", "Republic of Korea"],
    "392": ["JP", "Japan", "Япония", "Yaponiya"],
    "356": ["IN", "India", "Индия", "Hindiston"],
    "586": ["PK", "Pakistan", "Пакистан", "Pokiston"],
    "364": ["IR", "Iran", "Иран", "Eron"],
    "784": ["AE", "UAE", "ОАЭ", "BAA", "United Arab Emirates", "Эмираты"],
    "840": ["US", "USA", "США", "AQSH", "United States", "Америка"],
    "826": ["GB", "UK", "Великобритания", "Buyuk Britaniya", "England"],
    "250": ["FR", "France", "Франция", "Fransiya"],
    "380": ["IT", "Italy", "Италия", "Italiya"],
    "724": ["ES", "Spain", "Испания", "Ispaniya"],
    "616": ["PL", "Poland", "Польша", "Polsha"],
    "112": ["BY", "Belarus", "Беларусь", "Belorussiya", "Белоруссия"],
    "804": ["UA", "Ukraine", "Украина", "Ukraina"],
    "031": ["AZ", "Azerbaijan", "Азербайджан", "Ozarbayjon"],
    "051": ["AM", "Armenia", "Армения", "Armaniston"],
    "268": ["GE", "Georgia", "Грузия", "Gruziya"],
    "704": ["VN", "Vietnam", "Вьетнам", "Vyetnam"],
    "764": ["TH", "Thailand", "Таиланд", "Tailand"],
    "458": ["MY", "Malaysia", "Малайзия", "Malayziya"],
    "360": ["ID", "Indonesia", "Индонезия", "Indoneziya"],
    "528": ["NL", "Netherlands", "Нидерланды", "Niderlandiya", "Голландия"],
    "056": ["BE", "Belgium", "Бельгия", "Belgiya"],
    "756": ["CH", "Switzerland", "Швейцария", "Shveytsariya"],
    "348": ["HU", "Hungary", "Венгрия", "Vengriya"],
    "203": ["CZ", "Czech", "Чехия", "Chexiya"],
    "100": ["BG", "Bulgaria", "Болгария", "Bolgariya"],
    "642": ["RO", "Romania", "Румыния", "Ruminiya"],
    "428": ["LV", "Latvia", "Латвия", "Latviya"],
    "440": ["LT", "Lithuania", "Литва", "Litva"],
    "233": ["EE", "Estonia", "Эстония", "Estoniya"],
}

# --- O'lchov birligi klassifikatori (OKEI) ---
UNIT_CODE2NAMES: dict[str, list[str]] = {
    "796": ["шт", "шт.", "штука", "штук", "дона", "dona", "pcs", "piece", "pc"],
    "166": ["кг", "кг.", "kg", "килограмм", "kilogramm"],
    "168": ["т", "тонна", "tonna", "ton", "mt"],
    "006": ["м", "метр", "metr", "m"],
    "055": ["м2", "кв.м", "kv.m", "m2"],
    "113": ["м3", "куб.м", "kub.m", "m3"],
    "112": ["л", "литр", "litr", "l"],
    "715": ["пара", "пар", "juft", "pair"],
    "778": ["упак", "упаковка", "o'ram", "pack"],
    "736": ["рулон", "rulon", "roll"],
    "625": ["лист", "varaq", "sheet"],
    "704": ["набор", "to'plam", "set", "komplekt", "комплект"],
    "839": ["комплект", "komplekt", "set"],
}

# --- EISVO incoterms klassifikatori (rasmiy ro'yxat bo'yicha) ---
INCOTERMS_CODE2TERM = {
    "01": "EXW", "02": "FCA", "03": "FAS", "04": "FOB", "05": "CFR",
    "06": "CIF", "07": "CPT", "08": "CIP", "09": "DAF", "10": "DES",
    "11": "DEQ", "12": "DDU", "13": "DDP", "14": "DAP", "15": "DAT",
}

# eski atamalar bir xil ma'noda (DAT->DPU va h.k.) — deterministic.INCOTERMS_MAP
# normalizatsiyasi bilan birga ishlaydi
_INCOTERMS_EQUIV = {"DAT": "DPU", "DAF": "DAP", "DES": "DAP", "DEQ": "DPU", "DDU": "DAP"}


def currency_matches(pdf_value: str, api_code: str) -> bool | None:
    """None = kodni bilmaymiz (taqqoslab bo'lmaydi)."""
    alpha = CURRENCY_NUM2ALPHA.get(str(api_code).strip())
    if alpha is None:
        return None
    return str(pdf_value).strip().upper() == alpha


def country_matches(pdf_value: str, api_code: str) -> bool | None:
    names = COUNTRY_NUM2NAMES.get(str(api_code).strip().zfill(3))
    if names is None:
        return None
    v = str(pdf_value).strip().lower()
    return any(
        v == n.lower() or fuzz.partial_ratio(v, n.lower()) >= 90 for n in names
    )


def _unit_key(s: str) -> str:
    """O'lchov birligini solishtirishga tayyorlash: kichik harf + translit
    (кг->kg, шт->sht, м2->m2), bo'sh joy va oxirgi nuqtani olib tashlash."""
    return _translit(str(s).strip().lower()).replace(" ", "").rstrip(".")


def unit_matches(pdf_value: str, api_code: str) -> bool | None:
    names = UNIT_CODE2NAMES.get(str(api_code).strip().zfill(3))
    if names is None:
        return None
    targets = {_unit_key(n) for n in names}
    # bilingual / ko'p variantli qiymat: "кг/kg", "штук/ piece", "м2 /м2" —
    # ajratgichlar bo'yicha bo'lib, har bir qismni alohida tekshiramiz
    return any(
        _unit_key(part) in targets
        for part in re.split(r"[\/,;]", str(pdf_value))
        if part.strip()
    )


def incoterms_matches(pdf_value: str, api_code: str) -> bool | None:
    term = INCOTERMS_CODE2TERM.get(str(api_code).strip().zfill(2))
    if term is None:
        return None
    a = _INCOTERMS_EQUIV.get(term, term)
    b = str(pdf_value).strip().upper()
    b = _INCOTERMS_EQUIV.get(b, b)
    return a == b

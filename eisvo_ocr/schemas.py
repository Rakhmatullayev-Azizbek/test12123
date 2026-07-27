"""LLM chiqishi uchun Pydantic sxemalar — vLLM guided_json shulardan quriladi.

Bosqich A (META): seller/buyer/bank/delivery — butun matndan.
Bosqich B (PRODUCTS): faqat jadval markdown'laridan.

Eslatma: barcha maydonlar Optional — LLM topa olmasa null qaytaradi,
o'ylab topmasligi (hallucination) uchun promptda qat'iy aytiladi.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class BankInfo(BaseModel):
    name: str | None = Field(None, description="Bank nomi")
    address: str | None = None
    account: str | None = Field(None, description="Hisob raqami (20 raqam)")
    mfo: str | None = Field(None, description="MFO (5 raqam)")
    swift: str | None = Field(None, description="SWIFT/BIC kodi")


class Party(BaseModel):
    name: str | None = Field(None, description="Tashkilot to'liq nomi")
    country: str | None = None
    address: str | None = None
    inn: str | None = Field(None, description="INN/STIR (9 raqam)")
    phone: str | None = None
    director: str | None = Field(None, description="Imzolovchi shaxs (FIO)")
    # ataylab Optional EMAS: guided-generation'da null'ga ruxsat berilsa model
    # butun obyektni tashlab ketadi — obyekt majburiy, ichidagi maydonlar nullable
    bank: BankInfo = Field(default_factory=BankInfo)


class DeliveryTerms(BaseModel):
    incoterms: str | None = Field(None, description="Incoterms 2020 (EXW/FCA/.../DPU/DDP)")
    place: str | None = Field(None, description="Yetkazib berish joyi")
    deadline: str | None = Field(None, description="Muddat (matnda qanday bo'lsa)")


class ContractMeta(BaseModel):
    """Bosqich A natijasi."""
    contract_number: str | None = None
    contract_date: str | None = Field(None, description="ISO formatda: YYYY-MM-DD")
    seller: Party = Field(default_factory=Party)
    buyer: Party = Field(default_factory=Party)
    delivery: DeliveryTerms = Field(default_factory=DeliveryTerms)
    currency: str | None = Field(None, description="Valyuta kodi: USD/EUR/UZS/...")
    total_amount: float | None = Field(None, description="Shartnoma umumiy summasi")
    payment_terms: str | None = Field(None, description="To'lov shartlari (qisqacha)")


class Product(BaseModel):
    name: str | None = Field(None, description="Tovar nomi")
    hs_code: str | None = Field(None, description="TN VED / HS kodi, bo'lsa")
    unit: str | None = Field(None, description="O'lchov birligi: dona/kg/tonna/...")
    quantity: float | None = None
    price: float | None = Field(None, description="Birlik narxi")
    amount: float | None = Field(None, description="Qator summasi (quantity*price)")


class ProductList(BaseModel):
    """Bosqich B natijasi."""
    products: list[Product] = Field(default_factory=list)
    total_amount: float | None = Field(None, description="Jadvaldagi jami summa")


class ExtractedContract(BaseModel):
    """Yakuniy birlashgan natija (META + PRODUCTS)."""
    meta: ContractMeta
    products: ProductList

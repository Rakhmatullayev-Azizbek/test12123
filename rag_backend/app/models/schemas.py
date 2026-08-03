"""
API so'rov/javob modellari. Field(..., examples=[...]) qiymatlari Swagger UI
(/docs) da "Try it out" uchun tayyor namuna sifatida ko'rinadi.
"""
from pydantic import BaseModel, Field


# ---------- Chat ----------


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Foydalanuvchi savoli (o'zbek tilida).",
        examples=["AQShga xizmat safari uchun sutkalik xarajat qancha to'lanadi?"],
    )
    top_k: int = Field(
        8,
        ge=1,
        le=20,
        description=(
            "Qdrant'dan olinadigan eng mos chunk soni. Jadval qatorlari qisqa "
            "bo'lgani uchun kattaroq qiymat aniqlikni oshiradi."
        ),
    )
    use_history: bool = Field(
        True,
        description=(
            "True bo'lsa - shu sessiyaning oxirgi savol-javoblari ham LLM'ga "
            "kontekst sifatida beriladi (kuzatuv savollari uchun)."
        ),
    )


class SourceChunk(BaseModel):
    document_id: str = Field(..., examples=["-2818868"])
    document_title: str = Field(
        ...,
        examples=[
            "Davlat organlari va tashkilotlari xodimlari Oʻzbekiston Respublikasi "
            "tashqarisiga xizmat safariga yuborilganda..."
        ],
    )
    section_type: str | None = Field(
        None,
        description="Hujjat qismi turi: buyruq | nizom | kodeks | jadval",
        examples=["nizom"],
    )
    chapter: str | None = Field(None, description="Bob yoki jadval sarlavhasi.")
    band_number: str | None = Field(
        None, description="Band/modda raqami (aniqlangan bo'lsa).", examples=["54"]
    )
    score: float | None = Field(
        None, description="Qdrant o'xshashlik bahosi (0..1, kosinus).", examples=[0.89]
    )
    source_url: str | None = Field(
        None,
        description="lex.uz'dagi aniq bandga havola (element anchor bilan).",
        examples=["https://lex.uz/docs/-2818868#-7775523"],
    )


class ChatResponse(BaseModel):
    answer: str = Field(..., description="LLM javobi.")
    sources: list[SourceChunk] = Field(
        default_factory=list, description="Javob asoslangan hujjat bo'laklari."
    )
    session_id: str = Field(
        ..., description="Cookie orqali biriktirilgan sessiya identifikatori (UUID)."
    )
    retrieved_count: int = Field(
        0, description="Qdrant'dan olingan va filtrdan o'tgan chunk soni."
    )
    rewritten_question: str | None = Field(
        None,
        description=(
            "Kuzatuv savoli tarix asosida mustaqil savolga aylantirilgan bo'lsa - "
            "qidiruvda ishlatilgan to'liq savol. Aks holda null."
        ),
        examples=["Angliyaga bir kunlik safar xarajatlari qancha?"],
    )
    latency_ms: int = Field(0, description="So'rovni bajarish vaqti (millisekund).")


# ---------- Qidiruv (LLM'siz, debug uchun) ----------


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        examples=["texnologik asbob-uskunalar bojxona bojidan ozod"],
    )
    top_k: int = Field(10, ge=1, le=50)
    section_type: str | None = Field(
        None,
        description="Faqat shu turdagi bo'laklar bilan cheklash: buyruq | nizom | kodeks | jadval",
        examples=["jadval"],
    )


class SearchHit(BaseModel):
    score: float
    document_id: str
    document_title: str
    section_type: str | None = None
    chapter: str | None = None
    band_number: str | None = None
    source_url: str | None = None
    embed_text: str = Field(..., description="Vektorlashtirilgan matn bo'lagi.")


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


# ---------- Tarix ----------


class HistoryItem(BaseModel):
    question: str
    answer: str
    created_at: str


class HistoryResponse(BaseModel):
    session_id: str
    message_count: int = Field(0, description="Sessiyadagi jami savol soni.")
    first_seen_at: str | None = None
    last_seen_at: str | None = None
    items: list[HistoryItem] = Field(default_factory=list)


# ---------- Health / Stats ----------


class HealthResponse(BaseModel):
    status: str = Field(..., description="ok | degraded", examples=["ok"])
    qdrant_connected: bool
    postgres_connected: bool
    llm_reachable: bool
    collection_exists: bool
    indexed_chunks: int = Field(0, description="Qdrant kolleksiyasidagi point soni.")


class StatsResponse(BaseModel):
    collection: str
    indexed_chunks: int
    documents_total: int = 0
    documents_indexed: int = 0
    documents_pending: int = 0
    embedding_model: str
    llm_model: str


class ErrorResponse(BaseModel):
    detail: str = Field(..., examples=["Qdrant kolleksiyasi topilmadi - indexer ishga tushmagan."])

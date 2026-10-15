from typing import Literal
from pydantic import BaseModel, Field


class Document(BaseModel):
    content: str
    source: str
    page: int | None = None
    row: int | None = None
    document_type: str


class Chunk(Document):
    id: str
    document_id: str
    ordinal: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    limit: int = Field(default=5, ge=1, le=20)


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    session_id: str = Field(default="default", pattern=r"^[a-zA-Z0-9_-]{1,64}$")


class WebRequest(BaseModel):
    url: str = Field(max_length=2000)


class Evidence(BaseModel):
    id: str
    source: str
    text: str
    page: int | None = None
    row: int | None = None
    score: float
    url: str


class Claim(BaseModel):
    text: str
    evidence_ids: list[str]


class ChatResponse(BaseModel):
    answer: str
    claims: list[Claim] = []
    confidence: float = 0
    confidence_note: str = "Evidence coverage heuristic, not a calibrated probability."
    sources: list[Evidence] = []
    tools_used: list[str] = []
    status: Literal["answered", "abstained", "blocked"]
    mode: str
    cached: bool = False
    trace_id: str
    latency_ms: float = 0
    usage: dict = {}

from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question utilisateur")
    k: int = Field(6, ge=1, le=20, description="Nombre de chunks récupérés (top-k)")


class SourceItem(BaseModel):
    id: str
    title: str
    url: str
    city: str
    start: str
    end: str


class AskResponse(BaseModel):
    turn_id: str
    question: str
    answer: str
    sources: List[SourceItem]
    model: str
    k: int
    rating: Optional[Literal[-1, 1]] = None  # 👍=1, 👎=-1, None si pas noté


class HistoryItem(BaseModel):
    turn_id: str
    ts_utc: str
    question: str
    answer: str
    sources: List[SourceItem]
    model: str
    k: int
    rating: Optional[Literal[-1, 1]] = None


class HistoryResponse(BaseModel):
    items: List[HistoryItem]


class FeedbackRequest(BaseModel):
    turn_id: str = Field(..., min_length=1)
    vote: Literal[-1, 1]  # -1 = 👎, 1 = 👍


class FeedbackResponse(BaseModel):
    status: Literal["ok"]
    turn_id: str
    rating: Literal[-1, 1]


class RebuildResponse(BaseModel):
    status: str
    rows: int
    docs: int
    chunks: int
    index_dir: str

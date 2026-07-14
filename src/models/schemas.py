from pydantic import BaseModel
from typing import Optional


class ClothingItem(BaseModel):
    item: str
    color: Optional[str] = None


class ParsedQuery(BaseModel):
    clothing: list[ClothingItem] = []
    environment: Optional[str] = None
    style: Optional[str] = None
    activity: Optional[str] = None


class ImageMeta(BaseModel):
    path: str
    faiss_id: int


class RetrievalResult(BaseModel):
    path: str
    score: float
    clip_similarity: float
    atomic_similarity: float
    attr_consistency: float

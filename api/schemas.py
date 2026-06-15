from __future__ import annotations
from pydantic import BaseModel, Field


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")


class ReadyResponse(BaseModel):
    status: str = Field(..., example="ready")
    model_name: str = Field(..., example="my_model")
    model_version: str = Field(..., example="3")
    model_stage: str = Field(..., example="production")


# ── Inference ─────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, example="สินค้าดีมากครับ")


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        example=["สินค้าดีมากครับ", "แย่มากเลย", "ปกติธรรมดา"],
    )


class PredictResponse(BaseModel):
    label: str = Field(..., example="positive")
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.97)
    probabilities: dict[str, float] = Field(
        ...,
        example={"positive": 0.97, "neutral": 0.02, "negative": 0.01},
    )
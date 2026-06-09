from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


# ── Requests ───────────────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Thai text to classify.",
    )

    @field_validator("text")
    @classmethod
    def text_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty or whitespace only.")
        return v.strip()


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of Thai texts to classify (max 100 per call).",
    )

    @field_validator("texts")
    @classmethod
    def texts_not_empty(cls, v: list[str]) -> list[str]:
        for i, text in enumerate(v):
            if not text.strip():
                raise ValueError(f"texts[{i}] must not be empty or whitespace only.")
        return [t.strip() for t in v]


# ── Responses ──────────────────────────────────────────────────────────────────


class PredictResponse(BaseModel):
    text: str = Field(description="Original input text.")
    label: str = Field(description="Predicted sentiment label (neg/neu/pos/q).")
    label_id: int = Field(description="Integer label id (0=neg, 1=neu, 2=pos, 3=q).")
    confidence: float = Field(ge=0.0, le=1.0, description="Probability of the top label.")
    probabilities: dict[str, float] = Field(description="Softmax probability for each class.")
    model_version: str = Field(description="MLflow model registry version used.")


class HealthResponse(BaseModel):
    status: str = Field(description="'healthy' when the server is running.")


class ReadyResponse(BaseModel):
    status: str = Field(description="'ready' when the model is loaded.")
    model_name: str = Field(description="Registered model name in MLflow.")
    model_version: str = Field(description="Model version currently loaded.")
    model_stage: str = Field(description="Registry stage (Production).")
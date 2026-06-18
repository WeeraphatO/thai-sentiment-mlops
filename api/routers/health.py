from __future__ import annotations
from fastapi import APIRouter, HTTPException, Request
from api.schemas import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])

@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="healthy")

@router.get("/ready", response_model=ReadyResponse)
async def ready(request: Request) -> ReadyResponse:
    predictor = request.app.state.predictor
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Model is still loading.")
    return ReadyResponse(
        status="ready",
        model_name=predictor.model_name,
        model_version=predictor.model_version,
        model_stage="production",
    )
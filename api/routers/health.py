from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from src.serving.schemas import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    description="Returns 200 as long as the server process is running.",
)
async def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness check",
    description=(
        "Returns 200 once the Production model has finished loading from "
        "MLflow. Returns 503 while the model is still loading on startup."
    ),
)
async def ready(request: Request) -> ReadyResponse:
    predictor = request.app.state.predictor

    if not predictor.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Retry in a few seconds.",
        )

    return ReadyResponse(
        status="ready",
        model_name=predictor.model_name,
        model_version=predictor.model_version,
        model_stage="Production",
    )
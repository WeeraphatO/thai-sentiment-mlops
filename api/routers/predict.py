from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from src.serving.schemas import (
    BatchPredictRequest,
    PredictRequest,
    PredictResponse,
)

router = APIRouter(tags=["inference"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify a single Thai text",
    description=(
        "Returns the predicted sentiment label with confidence score "
        "and full class probabilities."
    ),
)
async def predict(
    body: PredictRequest,
    request: Request,
) -> PredictResponse:
    predictor = request.app.state.predictor

    if not predictor.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Check GET /ready and retry.",
        )

    result = predictor.predict(body.text)
    return PredictResponse(**result)


@router.post(
    "/predict/batch",
    response_model=list[PredictResponse],
    summary="Classify a batch of Thai texts",
    description=(
        "Accepts 1–100 texts per call. All texts are tokenized together "
        "in a single forward pass, making this more efficient than "
        "calling /predict in a loop."
    ),
)
async def predict_batch(
    body: BatchPredictRequest,
    request: Request,
) -> list[PredictResponse]:
    predictor = request.app.state.predictor

    if not predictor.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Check GET /ready and retry.",
        )

    results = predictor.predict_batch(body.texts)
    return [PredictResponse(**r) for r in results]
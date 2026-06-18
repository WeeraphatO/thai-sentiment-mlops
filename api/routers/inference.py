from __future__ import annotations
from fastapi import APIRouter, HTTPException, Request
from api.schemas import BatchPredictRequest, PredictRequest, PredictResponse

router = APIRouter(tags=["inference"])

@router.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, request: Request) -> PredictResponse:
    predictor = request.app.state.predictor
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Model is still loading.")
    result = predictor.predict(body.text)
    return PredictResponse(**result)

@router.post("/predict/batch", response_model=list[PredictResponse])
async def predict_batch(body: BatchPredictRequest, request: Request) -> list[PredictResponse]:
    predictor = request.app.state.predictor
    if not predictor.is_ready:
        raise HTTPException(status_code=503, detail="Model is still loading.")
    results = predictor.predict_batch(body.texts)
    return [PredictResponse(**r) for r in results]
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

from api.predictor import ModelPredictor
from api.routers.health import router as health_router
from api.routers.inference import router as inference_router
from src.utils.load_config import load_training_config

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Predictor Factory
# ------------------------------------------------------------------


def create_predictor() -> ModelPredictor:
    config = load_training_config()

    return ModelPredictor(
        model_name=config["mlflow"]["model_name"],
        tracking_uri=os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://localhost:5000",
        )
    )


# ------------------------------------------------------------------
# FastAPI Lifespan
# ------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting API...")

    predictor = create_predictor()

    predictor.load()

    # Optional: wait until model is loaded
    timeout_seconds = 120

    import time

    start = time.time()

    while not predictor.is_ready:
        if time.time() - start > timeout_seconds:
            raise RuntimeError(
                "Timed out while loading MLflow model."
            )

        time.sleep(1)

    logger.info(
        "Loaded model '%s' version '%s'",
        predictor.model_name,
        predictor.model_version,
    )

    app.state.predictor = predictor

    yield

    logger.info("Shutting down API...")


# ------------------------------------------------------------------
# Application
# ------------------------------------------------------------------

app = FastAPI(
    title="Thai Sentiment API",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(inference_router)
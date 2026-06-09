from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.middleware.logging import LoggingMiddleware
from api.routers import health, predict
from src.serving.predictor import SentimentPredictor
from src.utils.load_config import load_training_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
predictor = SentimentPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_training_config(
        os.getenv("CONFIG_PATH", "configs/train.yaml")
    )
    model_name = config["mlflow"]["model_name"]
    logger.info("Loading model '%s' from MLflow registry...", model_name)
    predictor.load_from_registry(model_name=model_name, stage="Production")
    app.state.predictor = predictor
    logger.info("API ready.")
    yield
    logger.info("Shutting down.")

app = FastAPI(
    title="Thai Sentiment Analysis API",
    description=(
        "Real-time sentiment classification for Thai text using PhayaThaiBERT. "
        "Labels: neg · neu · pos · q"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(LoggingMiddleware)
app.include_router(health.router)
app.include_router(predict.router, prefix="/api/v1")
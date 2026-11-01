"""FastAPI microservice serving the champion model for real-time inference."""
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas import HealthResponse, ModelInfo, PredictRequest, PredictResponse

logger = logging.getLogger("ml-api")
logging.basicConfig(level=logging.INFO)

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "model"))
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODEL_DIR / "champion_model.joblib")
    meta_path = MODEL_DIR / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    state["features"] = meta.get("feature_names") or list(state["model"].feature_names_in_)
    state["classes"] = meta.get("classes", {"0": "malignant", "1": "benign"})
    state["meta"] = meta
    logger.info("Model loaded (%d features)", len(state["features"]))
    yield
    state.clear()


app = FastAPI(
    title="ML Inference API",
    description="Real-time breast-cancer classification (Random Forest champion model).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return 422 with location + message only (echoing raw input can contain NaN, which is not valid JSON)."""
    detail = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": detail})


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded="model" in state)


@app.get("/model-info", response_model=ModelInfo, tags=["system"])
def model_info() -> ModelInfo:
    meta = state["meta"]
    return ModelInfo(
        model_type=meta.get("model_type", type(state["model"]).__name__),
        n_features=len(state["features"]),
        feature_names=state["features"],
        classes=state["classes"],
        cv_roc_auc=meta.get("cv_roc_auc"),
        test_roc_auc=meta.get("test_roc_auc"),
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest) -> PredictResponse:
    """Return the predicted class and class probabilities for one sample."""
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    X = pd.DataFrame([req.features], columns=state["features"])
    proba = model.predict_proba(X)[0]
    idx = int(proba.argmax())
    cls = int(model.classes_[idx])
    names = {str(int(c)): state["classes"][str(int(c))] for c in model.classes_}
    return PredictResponse(
        prediction=cls,
        label=names[str(cls)],
        confidence=round(float(proba[idx]), 6),
        probabilities={names[str(int(c))]: round(float(p), 6) for c, p in zip(model.classes_, proba)},
    )

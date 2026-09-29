"""Start with: python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000."""
import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from backend.model import ModelUnavailableError, SarcasmPredictor
from backend.type_model import SarcasmTypePredictor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor = SarcasmPredictor()
    app.state.predictor = predictor
    app.state.model_error = None
    try:
        predictor.load()
    except ModelUnavailableError as exc:
        app.state.model_error = str(exc)
        logger.warning("%s", exc)
    app.state.type_predictor = SarcasmTypePredictor()
    app.state.type_predictor.load()
    yield


app = FastAPI(title="SarcasmLens AI", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500", "http://127.0.0.1:3000", "http://localhost:3000"],
    allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
)


class Conversation(BaseModel):
    context: str = Field(min_length=1, max_length=20000)
    response: str = Field(min_length=1, max_length=10000)

    @field_validator("context", "response")
    @classmethod
    def strip_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Please enter text, not just spaces.")
        return value


class TypePrediction(BaseModel):
    status: Literal["predicted", "not_applicable", "unavailable"]
    code: str | None = None
    name: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    probabilities: dict[str, float] = Field(default_factory=dict)
    note: str


class Prediction(BaseModel):
    prediction: Literal["SARCASM", "NOT_SARCASM"]
    confidence: float = Field(ge=0, le=1)
    sarcasm_probability: float = Field(ge=0, le=1)
    not_sarcasm_probability: float = Field(ge=0, le=1)
    sarcasm_type: TypePrediction | None = None


class Comparison(BaseModel):
    response_only: Prediction
    context_aware: Prediction


def get_predictor(request: Request) -> SarcasmPredictor:
    predictor = request.app.state.predictor
    if not predictor.ready:
        raise HTTPException(status_code=503, detail=request.app.state.model_error or "Train the model first, then restart the backend.")
    return predictor


@app.get("/health")
def health(request: Request):
    predictor = request.app.state.predictor
    return {"status": "ok" if predictor.ready else "model_missing", "model_loaded": predictor.ready,
            "type_model_loaded": request.app.state.type_predictor.ready,
            "device": str(predictor.device), "message": "Model ready." if predictor.ready else request.app.state.model_error}


@app.get("/metrics")
def metrics(request: Request):
    return get_predictor(request).metrics


@app.get("/type-metrics")
def type_metrics(request: Request):
    predictor = request.app.state.type_predictor
    if not predictor.ready:
        raise HTTPException(status_code=503, detail=predictor.error)
    return predictor.metrics


def with_type(result, payload, request):
    try:
        result["sarcasm_type"] = request.app.state.type_predictor.predict(payload.context, payload.response, result["prediction"])
    except Exception:
        logger.exception("Sarcasm type inference failed")
        result["sarcasm_type"] = {"status": "unavailable", "note": "Sarcasm-type inference is temporarily unavailable."}
    return result


def run_prediction(predictor, response, context=None):
    try:
        return predictor.predict(response=response, context=context)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="The model could not complete this analysis. Try shorter text or restart the backend.") from exc


@app.post("/predict", response_model=Prediction)
def predict(payload: Conversation, request: Request):
    return with_type(run_prediction(get_predictor(request), payload.response, payload.context), payload, request)


@app.post("/compare", response_model=Comparison)
def compare(payload: Conversation, request: Request):
    predictor = get_predictor(request)
    return {"response_only": run_prediction(predictor, payload.response),
            "context_aware": with_type(run_prediction(predictor, payload.response, payload.context), payload, request)}

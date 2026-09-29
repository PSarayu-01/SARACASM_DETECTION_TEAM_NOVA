"""Shared configuration and inference. Never falls back to an untrained model."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "saved_model"
DATA_PATH = ROOT / "data" / "mustard_context_sarcasm_cleaned.csv"
OUTPUT_DIR = ROOT / "outputs"
BASE_MODEL = "FacebookAI/roberta-base"
MAX_LENGTH = 256
ID2LABEL = {0: "NOT_SARCASM", 1: "SARCASM"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


class ModelUnavailableError(RuntimeError):
    """The fine-tuned local checkpoint is missing or invalid."""


class SarcasmPredictor:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.metrics = None
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def load(self) -> None:
        """Called once at API startup; all inference loads are strictly local."""
        if not (self.model_dir / "metrics.json").is_file():
            raise ModelUnavailableError("Train the model first with python -m backend.train, then restart the backend.")
        try:
            metrics = json.loads((self.model_dir / "metrics.json").read_text(encoding="utf-8"))
            if metrics.get("training_complete") is not True or metrics.get("max_length") != MAX_LENGTH:
                raise ValueError("Missing or incompatible completed training metadata.")
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_dir, local_files_only=True)
            if model.config.num_labels != 2 or model.config.id2label != ID2LABEL:
                raise ValueError("Checkpoint label mapping must be 0=NOT_SARCASM, 1=SARCASM.")
            model.to(self.device)
            model.eval()
        except Exception as exc:
            raise ModelUnavailableError("The saved model could not be loaded. Complete training and restart the backend.") from exc
        self.tokenizer, self.model, self.metrics = tokenizer, model, metrics

    def predict(self, response: str, context: str | None = None) -> dict:
        if not self.ready:
            raise ModelUnavailableError("Train the model first with python -m backend.train, then restart the backend.")
        # Training and context-aware inference use exactly the same text-pair format.
        # Response-only is an ablation of this model, not a separately trained baseline.
        with self._lock, torch.inference_mode():
            if context is None:
                encoded = self.tokenizer(response, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            else:
                encoded = self.tokenizer(context, response, truncation="longest_first", max_length=MAX_LENGTH, return_tensors="pt")
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            probabilities = torch.softmax(self.model(**encoded).logits, dim=-1)[0].cpu().tolist()
        predicted = int(probabilities[1] > probabilities[0])
        return {
            "prediction": ID2LABEL[predicted],
            "confidence": probabilities[predicted],
            "sarcasm_probability": probabilities[1],
            "not_sarcasm_probability": probabilities[0],
        }

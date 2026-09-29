"""Independent, context-pair RoBERTa features with a trained softmax type head."""
import json
import threading
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

from backend.model import ROOT, MAX_LENGTH

TYPE_DIR = ROOT / "saved_type_model"
TYPE_NAMES = {"PRO": "Propositional", "ILL": "Illocutionary", "EMB": "Embedded"}
TYPE_NOTE = "Text-only estimate among three supported types. Like-prefixed sarcasm is unsupported: only 3 labelled examples."


@torch.inference_mode()
def pair_features(encoder, tokenizer, contexts, responses, device):
    inputs = tokenizer(contexts, responses, padding=True, truncation="longest_first",
                       max_length=MAX_LENGTH, return_tensors="pt").to(device)
    states = encoder(**inputs).last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1).to(states.dtype)
    return (states * mask).sum(1) / mask.sum(1).clamp_min(1)


class SarcasmTypePredictor:
    def __init__(self, model_dir=TYPE_DIR):
        self.model_dir = Path(model_dir)
        self.ready = False
        self.metrics = None
        self.error = "Train the sarcasm-type model with python -m backend.train_types, then restart the backend."
        self.lock = threading.Lock()

    def load(self):
        if not (self.model_dir / "metrics.json").exists():
            return
        try:
            metrics = json.loads((self.model_dir / "metrics.json").read_text(encoding="utf-8"))
            if not metrics.get("training_complete") or metrics["max_length"] != MAX_LENGTH:
                raise ValueError("Incomplete or incompatible type checkpoint")
            self.codes = metrics["label_order"]
            if self.codes != list(TYPE_NAMES):
                raise ValueError("Unexpected type labels")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir / "encoder", local_files_only=True)
            self.encoder = AutoModel.from_pretrained(self.model_dir / "encoder", local_files_only=True).to(self.device).eval()
            state = load_file(str(self.model_dir / "head.safetensors"), device=str(self.device))
            self.weight, self.bias = state["weight"], state["bias"]
            self.mean, self.scale = state["mean"], state["scale"]
            self.metrics = metrics
            self.ready = True
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Could not load sarcasm-type checkpoint")
            self.error = "The sarcasm-type checkpoint could not be loaded. Retrain it and restart the backend."

    def predict(self, context, response, binary_prediction):
        if binary_prediction != "SARCASM":
            return {"status": "not_applicable", "code": None, "name": None, "confidence": None,
                    "probabilities": {}, "note": "No sarcasm type is assigned because the binary model predicted NOT_SARCASM."}
        if not self.ready:
            return {"status": "unavailable", "code": None, "name": None, "confidence": None,
                    "probabilities": {}, "note": self.error}
        with self.lock, torch.inference_mode():
            features = pair_features(self.encoder, self.tokenizer, [context], [response], self.device)
            logits = ((features - self.mean) / self.scale) @ self.weight.T + self.bias
            probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        index = max(range(len(probabilities)), key=probabilities.__getitem__)
        code = self.codes[index]
        return {"status": "predicted", "code": code, "name": TYPE_NAMES[code], "confidence": probabilities[index],
                "probabilities": dict(zip(self.codes, probabilities)), "note": TYPE_NOTE}

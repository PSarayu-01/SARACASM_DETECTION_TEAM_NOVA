"""Isolated plumbing tests; no test output is used as model-performance data."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from sklearn.model_selection import train_test_split

from backend import app as api_module
from backend.model import DATA_PATH, ID2LABEL, MAX_LENGTH, ModelUnavailableError, SarcasmPredictor
from backend.train import ConversationDataset, calculate_metrics, load_data


def test_supplied_dataset_and_stratified_split():
    frame = load_data(DATA_PATH)
    train, val = train_test_split(frame, test_size=0.2, stratify=frame.label, random_state=42)
    assert len(train) == 961 and len(val) == 241
    assert set(train.index).isdisjoint(val.index)
    assert sorted(val.label.value_counts().tolist()) == [120, 121]


def test_metric_definitions():
    scores = calculate_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert scores["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert scores["sarcasm_f1"] == pytest.approx(0.8)
    assert scores["accuracy"] == 0.75
    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["recall"] == 1.0
    assert scores["confusion_matrix"] == [[1, 1], [0, 2]]


def test_only_context_and_response_are_tokenized():
    frame = load_data(DATA_PATH).iloc[:2]
    calls = []

    def tokenizer(*args, **kwargs):
        calls.append((args, kwargs))
        return {"input_ids": [[0, 2], [0, 2]], "attention_mask": [[1, 1], [1, 1]]}

    dataset = ConversationDataset(frame, tokenizer)
    args, kwargs = calls[0]
    assert args == (frame.context.tolist(), frame.response.tolist())
    assert kwargs["max_length"] == 256
    assert dataset[0]["labels"] == frame.label.iloc[0]


def test_missing_checkpoint_never_loads_base_model(tmp_path):
    predictor = SarcasmPredictor(tmp_path)
    with pytest.raises(ModelUnavailableError):
        predictor.load()
    assert not predictor.ready
    with pytest.raises(ModelUnavailableError):
        predictor.predict("A response", "Some context")


def test_softmax_and_encoding_paths():
    predictor = SarcasmPredictor()
    predictor.device = torch.device("cpu")
    calls = []

    def tokenizer(*args, **kwargs):
        calls.append((args, kwargs))
        return {"input_ids": torch.tensor([[0, 2]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}

    class LogitModel:
        def __call__(self, **kwargs):
            assert not torch.is_grad_enabled()
            return SimpleNamespace(logits=torch.tensor([[2.0, -1.0]]))

    predictor.tokenizer = tokenizer
    predictor.model = LogitModel()
    result = predictor.predict("reply", "context")
    expected = torch.softmax(torch.tensor([2.0, -1.0]), dim=-1).tolist()
    assert result["prediction"] == ID2LABEL[0]
    assert result["confidence"] == pytest.approx(expected[0])
    assert result["sarcasm_probability"] == pytest.approx(expected[1])
    assert result["not_sarcasm_probability"] == pytest.approx(expected[0])
    assert calls[0][0] == ("context", "reply")
    assert calls[0][1]["max_length"] == MAX_LENGTH
    predictor.predict("reply")
    assert calls[1][0] == ("reply",)


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api_module, "SarcasmPredictor", lambda: SarcasmPredictor(tmp_path))
    with TestClient(api_module.app) as instance:
        yield instance


def test_untrained_api_and_validation(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is False
    assert client.get("/metrics").status_code == 503
    payload = {"context": "Something happened.", "response": "Really?"}
    for path in ("/predict", "/compare"):
        assert client.post(path, json=payload).status_code == 503
        for field in ("context", "response"):
            assert client.post(path, json={**payload, field: "   "}).status_code == 422
        assert client.post(path, json={"context": "Some context"}).status_code == 422
        assert client.post(path, json={**payload, "response": "a" * 10001}).status_code == 422


def test_cors_is_limited_to_local_frontend(client):
    headers = {"Origin": "http://127.0.0.1:5500", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"}
    response = client.options("/compare", headers=headers)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == headers["Origin"]
    response = client.options("/compare", headers={**headers, "Origin": "https://example.org"})
    assert "access-control-allow-origin" not in response.headers


def test_prediction_and_comparison_use_predictor(client):
    calls = []

    class Predictor:
        ready = True
        device = "cpu"
        metrics = {"test_only": True}

        def predict(self, response, context=None):
            calls.append((response, context))
            probabilities = torch.softmax(torch.tensor([0.1, 0.8]), dim=-1).tolist()
            return {"prediction": "SARCASM", "confidence": probabilities[1],
                    "sarcasm_probability": probabilities[1], "not_sarcasm_probability": probabilities[0]}

    api_module.app.state.predictor = Predictor()
    payload = {"context": " context ", "response": " reply "}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert np.isclose(result["sarcasm_probability"] + result["not_sarcasm_probability"], 1)
    assert calls == [("reply", "context")]
    calls.clear()
    response = client.post("/compare", json=payload)
    assert response.status_code == 200
    assert set(response.json()) == {"response_only", "context_aware"}
    assert calls == [("reply", None), ("reply", "context")]


def test_inference_failure_is_friendly(client):
    class BrokenPredictor:
        ready = True

        def predict(self, **kwargs):
            raise RuntimeError("Internal implementation detail")

    api_module.app.state.predictor = BrokenPredictor()
    response = client.post("/predict", json={"context": "Context", "response": "Reply"})
    assert response.status_code == 500
    assert "Internal implementation detail" not in response.text

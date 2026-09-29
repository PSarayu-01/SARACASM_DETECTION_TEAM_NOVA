import torch

from backend.type_model import SarcasmTypePredictor, TYPE_NAMES
from backend.train_types import prepare_data


def test_type_target_is_separate_and_split_disjoint():
    frame, train, validation = prepare_data()
    assert len(frame) == 598
    assert set(frame.sarcasm_type) == set(TYPE_NAMES)
    assert set(train).isdisjoint(validation)
    assert set(frame.iloc[train].pair).isdisjoint(frame.iloc[validation].pair)
    assert len(validation) == 120


def test_no_type_for_non_sarcasm_or_missing_checkpoint(tmp_path):
    predictor = SarcasmTypePredictor(tmp_path)
    predictor.load()
    assert predictor.predict("Context", "Response", "NOT_SARCASM")["status"] == "not_applicable"
    result = predictor.predict("Context", "Response", "SARCASM")
    assert result["status"] == "unavailable"
    assert result["confidence"] is None


def test_type_softmax_uses_both_inputs(monkeypatch):
    from backend import type_model
    seen = []
    def features(encoder, tokenizer, contexts, responses, device):
        seen.append((contexts, responses))
        return torch.tensor([[1., 2.]])
    monkeypatch.setattr(type_model, "pair_features", features)
    predictor = SarcasmTypePredictor()
    predictor.ready = True
    predictor.encoder = predictor.tokenizer = None
    predictor.device = "cpu"
    predictor.codes = list(TYPE_NAMES)
    predictor.mean = torch.zeros(2)
    predictor.scale = torch.ones(2)
    predictor.weight = torch.tensor([[1., 0.], [0., 1.], [-1., -1.]])
    predictor.bias = torch.zeros(3)
    result = predictor.predict("Context", "Response", "SARCASM")
    assert seen == [(["Context"], ["Response"])]
    assert result["code"] == "ILL"
    assert abs(sum(result["probabilities"].values())-1) < 1e-6
    assert result["confidence"] == max(result["probabilities"].values())

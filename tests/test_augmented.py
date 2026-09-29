import pandas as pd
import pytest

from backend.train_augmented import prepare_synthetic


def synthetic_rows():
    return [{"parent_context": f"Context {label}-{group}", "full_dialogue_context": f"Context {label}-{group}",
             "response": f"Response {variant}", "label": label}
            for label in (0, 1) for group in range(5) for variant in range(2)]


def test_duplicates_and_context_groups_cannot_cross_splits(tmp_path):
    rows = synthetic_rows()
    path = tmp_path / "synthetic.csv"
    pd.DataFrame(rows + rows).to_csv(path, index=False)
    original = pd.DataFrame({"context": ["Original"], "response": ["Reply"]})
    train, val, audit = prepare_synthetic(path, original, original)
    assert audit["duplicates_removed"] == 20
    assert len(train) + len(val) == 20
    assert set(train.group).isdisjoint(val.group)
    assert set(train.pair).isdisjoint(val.pair)
    assert set(train.label) == set(val.label) == {0, 1}


def test_original_validation_contexts_are_excluded(tmp_path):
    path = tmp_path / "synthetic.csv"
    pd.DataFrame(synthetic_rows()).to_csv(path, index=False)
    original = pd.DataFrame({"context": ["context 0-0"], "response": ["Other reply"]})
    train, val, audit = prepare_synthetic(path, original, original)
    assert "context 0-0" not in set(train.group) | set(val.group)
    assert audit["original_overlap_rows_removed"] == 2


def test_conflicting_duplicates_are_rejected(tmp_path):
    rows = synthetic_rows()
    rows.append({**rows[0], "label": 1})
    path = tmp_path / "synthetic.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    original = pd.DataFrame({"context": ["Original"], "response": ["Reply"]})
    with pytest.raises(ValueError, match="conflicting labels"):
        prepare_synthetic(path, original, original)

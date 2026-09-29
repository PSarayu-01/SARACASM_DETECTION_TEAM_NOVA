"""Train a separate type head on frozen pretrained RoBERTa text-pair features.

No binary-model weights or predictions enter this training. Only context and
response enter the encoder; sarcasm_type is the supervised target.
"""
import argparse
import hashlib
import json
import random
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

from backend.model import DATA_PATH, BASE_MODEL, MAX_LENGTH, OUTPUT_DIR
from backend.type_model import TYPE_DIR, TYPE_NAMES, TYPE_NOTE, pair_features


def prepare_data(path=DATA_PATH):
    frame = pd.read_csv(path, keep_default_na=False)
    for column in ("context", "response", "sarcasm_type"):
        frame[column] = frame[column].astype(str).str.strip()
    frame["label"] = pd.to_numeric(frame["label"], errors="raise")
    selected = frame[(frame.label == 1) & frame.sarcasm_type.isin(TYPE_NAMES)].copy()
    if selected.context.eq("").any() or selected.response.eq("").any():
        raise ValueError("Empty type-training text")
    selected["pair"] = list(zip(selected.context.str.casefold(), selected.response.str.casefold()))
    if (selected.groupby("pair").sarcasm_type.nunique() > 1).any():
        raise ValueError("Conflicting type labels for identical text pairs")
    selected = selected.drop_duplicates("pair")
    counts = selected.sarcasm_type.value_counts()
    if set(counts.index) != set(TYPE_NAMES) or counts.min() < 10:
        raise ValueError("Not enough examples of all three supported sarcasm types")
    train, validation = train_test_split(np.arange(len(selected)), test_size=0.2,
                                        random_state=42, stratify=selected.sarcasm_type)
    return selected, train, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100, help="Head-only epochs; RoBERTa remains frozen")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("Positive epochs and batch size required")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(4)
    frame, train_idx, val_idx = prepare_data()
    codes = list(TYPE_NAMES)
    labels = torch.tensor([codes.index(code) for code in frame.sarcasm_type], dtype=torch.long)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use the cached, original pretrained encoder, not the binary fine-tuned model.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    encoder = AutoModel.from_pretrained(BASE_MODEL, local_files_only=True).to(device).eval()
    encoder.requires_grad_(False)
    features = []
    for start in range(0, len(frame), args.batch_size):
        batch = frame.iloc[start:start+args.batch_size]
        features.append(pair_features(encoder, tokenizer, batch.context.tolist(), batch.response.tolist(), device).cpu())
        if start % 40 == 0:
            print(f"Encoding {min(start+args.batch_size,len(frame))}/{len(frame)} context-response pairs", flush=True)
    x = torch.cat(features)
    # Fit normalization on training rows only.
    mean, scale = x[train_idx].mean(0), x[train_idx].std(0).clamp_min(1e-5)
    x = (x-mean)/scale
    head = torch.nn.Linear(x.shape[1], len(codes))
    counts = torch.bincount(labels[train_idx], minlength=len(codes)).float()
    criterion = torch.nn.CrossEntropyLoss(weight=counts.sum()/(len(codes)*counts))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.003, weight_decay=0.1)
    best_score, best_state, best_epoch = -1.0, None, None
    history = []
    for epoch in range(1, args.epochs+1):
        head.train()
        order = torch.tensor(train_idx)[torch.randperm(len(train_idx))]
        for batch_idx in order.split(64):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(x[batch_idx]), labels[batch_idx])
            loss.backward(); optimizer.step()
        head.eval()
        with torch.inference_mode():
            predictions = head(x[val_idx]).argmax(-1).tolist()
        score = f1_score(labels[val_idx].tolist(), predictions, labels=list(range(3)), average="macro", zero_division=0)
        history.append({"epoch": epoch, "macro_f1": float(score)})
        if score > best_score:
            best_score, best_epoch = float(score), epoch
            best_state = {k: v.detach().clone() for k,v in head.state_dict().items()}
    head.load_state_dict(best_state)
    with torch.inference_mode():
        probs = head(x[val_idx]).softmax(-1)
    predicted, truth = probs.argmax(-1).tolist(), labels[val_idx].tolist()
    majority = int(counts.argmax())
    run_dir = OUTPUT_DIR / ("types_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_dir.mkdir(parents=True)
    report = {"training_complete": True, "model_kind": "Frozen pretrained RoBERTa mean-pool features + trained linear softmax head",
              "base_model": BASE_MODEL, "encoder_revision": getattr(encoder.config, "_commit_hash", None),
              "label_order": codes, "label_names": TYPE_NAMES, "max_length": MAX_LENGTH,
              "features": ["context", "response"], "target": "sarcasm_type", "seed": 42,
              "train_size": len(train_idx), "validation_size": len(val_idx), "class_counts": frame.sarcasm_type.value_counts().to_dict(),
              "macro_f1": best_score, "accuracy": float(accuracy_score(truth, predicted)), "best_epoch": best_epoch,
              "head_epochs": args.epochs, "head_learning_rate": 0.003, "weight_decay": 0.1,
              "majority_macro_f1": float(f1_score(truth, [majority]*len(truth), labels=list(range(3)), average="macro", zero_division=0)),
              "confusion_matrix": confusion_matrix(truth, predicted, labels=list(range(3))).tolist(),
              "per_class": classification_report(truth, predicted, labels=list(range(3)), target_names=codes, output_dict=True, zero_division=0),
              "dataset_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
              "note": TYPE_NOTE, "evaluation_note": "Conditional type validation on annotated sarcastic examples; not end-to-end binary+type accuracy. Validation selects the best head epoch, not an independent test score.",
              "completed_at": datetime.now(timezone.utc).isoformat(), "run_id": run_dir.name}
    TYPE_DIR.mkdir(parents=True, exist_ok=True)
    (TYPE_DIR / "metrics.json").unlink(missing_ok=True)
    encoder.save_pretrained(TYPE_DIR / "encoder", safe_serialization=True)
    tokenizer.save_pretrained(TYPE_DIR / "encoder")
    save_file({**best_state, "mean": mean, "scale": scale}, str(TYPE_DIR / "head.safetensors"))
    (run_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    frame.iloc[train_idx].drop(columns="pair").to_csv(run_dir / "train_split.csv", index_label="source_row")
    output = frame.iloc[val_idx].drop(columns="pair").copy()
    output["predicted_type"] = [codes[i] for i in predicted]
    for index, code in enumerate(codes): output["probability_"+code] = probs[:,index].tolist()
    output.to_csv(run_dir / "validation_predictions.csv", index_label="source_row")
    for path in (run_dir / "metrics.json", TYPE_DIR / "metrics.json"):
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

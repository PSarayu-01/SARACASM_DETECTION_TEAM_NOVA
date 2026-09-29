"""Continue local RoBERTa training with deduplicated, context-grouped synthetic data.

Run: python -m backend.train_augmented --epochs 3 --threads 4
The existing checkpoint is never overwritten by this script. An accepted candidate
is saved separately for publication after the backend has stopped.
"""
from __future__ import annotations

import argparse
import math
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

from backend.model import DATA_PATH, MODEL_DIR, OUTPUT_DIR, MAX_LENGTH, ID2LABEL
from backend.train import ConversationDataset, evaluate, load_data, seed_everything, write_json

SYNTHETIC_PATH = DATA_PATH.parent / "sarcasmlens_synthetic_2000_chatgpt.csv"


def normalized(text):
    return " ".join(str(text).casefold().split())


def prepare_synthetic(path, original, original_validation, seed=42):
    raw = pd.read_csv(path, keep_default_na=False)
    required = {"parent_context", "full_dialogue_context", "response", "label"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Synthetic file needs columns: {sorted(required)}")
    frame = pd.DataFrame({"context": raw.full_dialogue_context.astype(str).str.strip(),
                          "response": raw.response.astype(str).str.strip(),
                          "label": pd.to_numeric(raw.label, errors="coerce"),
                          "group": raw.parent_context.map(normalized),
                          "source_row": raw.index})
    if not frame.label.isin([0, 1]).all():
        raise ValueError("Synthetic labels must be 0 or 1.")
    frame.label = frame.label.astype(int)
    if frame.context.eq("").any() or frame.response.eq("").any() or frame.group.eq("").any():
        raise ValueError("Synthetic contexts, parent contexts, and responses must be nonempty.")
    frame["pair"] = list(zip(frame.context.map(normalized), frame.response.map(normalized)))
    if (frame.groupby("pair").label.nunique() > 1).any():
        raise ValueError("Identical synthetic text pairs have conflicting labels.")
    # The supplied file repeats the same full context under one parent context.
    # Reject ambiguous group mappings instead of allowing cross-split leakage.
    if (frame.groupby(frame.context.map(normalized)).group.nunique() > 1).any():
        raise ValueError("One full context maps to several parent groups; merge the groups first.")
    unique = frame.drop_duplicates("pair").copy()
    audit = {"input_rows": len(frame), "unique_pairs": len(unique), "duplicates_removed": len(frame)-len(unique)}
    original_pairs = set(zip(original.context.map(normalized), original.response.map(normalized)))
    val_contexts = set(original_validation.context.map(normalized))
    excluded_groups = set(unique.loc[unique.context.map(normalized).isin(val_contexts) | unique.group.isin(val_contexts), "group"])
    clean = unique.loc[~unique.pair.isin(original_pairs) & ~unique.group.isin(excluded_groups)].copy()
    audit["original_overlap_rows_removed"] = len(unique)-len(clean)
    # Stratify context groups, not the individual near-duplicate variants.
    signatures = clean.groupby("group").label.agg(lambda values: ",".join(map(str, sorted(set(values)))))
    if len(signatures) < 4 or signatures.value_counts().min() < 2:
        raise ValueError("Not enough independent context groups for a stratified synthetic holdout.")
    train_groups, val_groups = train_test_split(signatures.index, test_size=0.2, random_state=seed, stratify=signatures)
    train = clean.loc[clean.group.isin(train_groups)].copy()
    val = clean.loc[clean.group.isin(val_groups)].copy()
    assert set(train.group).isdisjoint(val.group)
    assert set(train.pair).isdisjoint(val.pair)
    audit.update({"training_rows": len(train), "validation_rows": len(val), "training_context_groups": len(train_groups),
                  "validation_context_groups": len(val_groups), "training_labels": train.label.value_counts().to_dict(),
                  "validation_labels": val.label.value_counts().to_dict(), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()})
    return train, val, audit


def prepare(args):
    baseline = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
    if not baseline.get("training_complete"):
        raise ValueError("A completed original checkpoint is required.")
    original = load_data(DATA_PATH)
    if hashlib.sha256(DATA_PATH.read_bytes()).hexdigest() != baseline["dataset_sha256"]:
        raise ValueError("Original dataset has changed since the saved checkpoint was trained.")
    split_run = baseline.get("original_split_run_id", baseline["run_id"])
    split_dir = OUTPUT_DIR / split_run
    train_rows = pd.read_csv(split_dir / "train_split.csv").source_row.tolist()
    val_rows = pd.read_csv(split_dir / "validation_split.csv").source_row.tolist()
    if len(set(train_rows + val_rows)) != len(original) or set(train_rows) & set(val_rows):
        raise ValueError("Original split manifest is incomplete or overlapping.")
    train_original, val_original = original.loc[train_rows].copy(), original.loc[val_rows].copy()
    syn_train, syn_val, audit = prepare_synthetic(args.synthetic, original, val_original, args.seed)
    train_original["source"] = "original"
    train_original["source_row"] = train_original.index
    syn_train["source"] = "synthetic"
    combined = pd.concat([train_original, syn_train], ignore_index=True)[["context", "response", "label", "source", "source_row"]]
    return baseline, split_run, combined, val_original, syn_val, audit


def run(args):
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    baseline, split_run, train_frame, original_val, synthetic_val, audit = prepare(args)
    print(json.dumps({"synthetic_audit": audit, "combined_training_rows": len(train_frame), "original_validation_rows": len(original_val)}, indent=2), flush=True)
    if args.prepare_only:
        return
    run_id = "augmented_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    write_json(OUTPUT_DIR / "latest_augmented_run.json", {"run_id": run_id, "run_directory": str(run_dir)})
    write_json(run_dir / "data_audit.json", audit)
    write_json(run_dir / "baseline_saved_metrics.json", baseline)
    train_frame.to_csv(run_dir / "training_data.csv", index=False)
    original_val[["scene", "key", "label"]].to_csv(run_dir / "original_validation_split.csv", index_label="source_row")
    synthetic_val.drop(columns="pair").to_csv(run_dir / "synthetic_validation_data.csv", index=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True).to(device)
    if model.config.id2label != ID2LABEL:
        raise ValueError("Unexpected checkpoint label mapping.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    collator = DataCollatorWithPadding(tokenizer)
    def loader(frame, shuffle=False):
        return DataLoader(ConversationDataset(frame, tokenizer), batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=collator, num_workers=0, generator=torch.Generator().manual_seed(args.seed))
    train_loader = loader(train_frame, True)
    original_loader, synthetic_loader = loader(original_val), loader(synthetic_val)
    baseline_original, _, _ = evaluate(model, original_loader, device)
    baseline_synthetic, _, _ = evaluate(model, synthetic_loader, device)
    write_json(run_dir / "baseline_evaluation.json", {"original_validation": baseline_original, "synthetic_validation": baseline_synthetic})
    print("Baseline original validation: " + json.dumps(baseline_original), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_loader)/args.accumulation_steps)
    total_steps = steps_per_epoch*args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps*0.1), total_steps)
    best_macro = -1.0
    accepted_macro = baseline_original["macro_f1"]
    history, best, accepted = [], None, None
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        step = 0
        for micro_step, batch in enumerate(train_loader, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss; stopping without modifying the existing model.")
            block_start = ((micro_step-1)//args.accumulation_steps)*args.accumulation_steps*args.batch_size
            block_size = min(args.batch_size*args.accumulation_steps, len(train_frame)-block_start)
            (loss * batch["labels"].size(0) / block_size).backward()
            total_loss += loss.item()*batch["labels"].size(0)
            if micro_step % args.accumulation_steps and micro_step != len(train_loader):
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step == 1 or step % 5 == 0 or step == steps_per_epoch:
                progress = {"status": "training", "epoch": epoch, "epochs": args.epochs, "step": step,
                            "steps_per_epoch": steps_per_epoch, "elapsed_seconds": round(time.monotonic()-started, 1)}
                write_json(run_dir / "progress.json", progress)
                print(json.dumps(progress), flush=True)
        original_metrics, predictions, probabilities = evaluate(model, original_loader, device)
        synthetic_metrics, _, _ = evaluate(model, synthetic_loader, device)
        record = {"epoch": epoch, "training_loss": total_loss/len(train_frame), "original_validation": original_metrics,
                  "synthetic_validation": synthetic_metrics}
        history.append(record)
        write_json(run_dir / "training_history.json", history)
        print("Validation: " + json.dumps(record), flush=True)
        report = {**baseline, **original_metrics, "run_id": run_id, "original_split_run_id": split_run,
                  "best_epoch": epoch, "epochs": args.epochs, "seed": args.seed, "learning_rate": args.learning_rate,
                  "batch_size": args.batch_size, "gradient_accumulation_steps": args.accumulation_steps,
                  "effective_batch_size": args.batch_size*args.accumulation_steps,
                  "train_size": len(train_frame), "validation_size": len(original_val),
                  "device": str(device), "synthetic_data": audit, "synthetic_validation": synthetic_metrics,
                  "initial_checkpoint_run_id": baseline["run_id"], "training_complete": True,
                  "completed_at": datetime.now(timezone.utc).isoformat(),
                  "split": "Original validation unchanged; synthetic 80/20 split by parent-context groups after deduplication",
                  "evaluation_note": "Original validation selects the checkpoint, not an independent test. Synthetic holdout is reported separately; its templates may still share style. Continued training from the previous checkpoint."}
        is_best = original_metrics["macro_f1"] > best_macro
        qualifies = (original_metrics["macro_f1"] > accepted_macro and
                     original_metrics["confusion_matrix"][0][1] <= baseline_original["confusion_matrix"][0][1])
        if is_best or qualifies:
            checkpoint = run_dir / f"checkpoint_epoch_{epoch}"
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            write_json(checkpoint / "metrics.json", report)
            output = original_val[["scene", "key", "label"]].copy()
            output["prediction"] = predictions
            output["sarcasm_probability"] = [p[1] for p in probabilities]
            output.to_csv(run_dir / f"validation_predictions_epoch_{epoch}.csv", index_label="source_row")
            candidate = {"checkpoint": str(checkpoint), "epoch": epoch, "metrics": original_metrics}
            if is_best:
                best_macro, best = original_metrics["macro_f1"], candidate
            if qualifies:
                accepted_macro, accepted = original_metrics["macro_f1"], candidate
    decision = {"baseline_original_validation": baseline_original, "baseline_synthetic_validation": baseline_synthetic,
                "best_by_macro_f1": best, "accepted_candidate": accepted,
                "acceptance_rule": "Higher original-validation Macro F1 and no more original-validation false positives than baseline.",
                "published": False, "note": "Existing saved_model is unchanged. Publish the accepted candidate after stopping the backend."}
    write_json(run_dir / "comparison.json", decision)
    write_json(run_dir / "progress.json", {"status": "complete", "elapsed_seconds": round(time.monotonic()-started, 1),
                                          "accepted_candidate": accepted["checkpoint"] if accepted else None})
    print("COMPLETE: " + json.dumps(decision), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, default=SYNTHETIC_PATH)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads, args.accumulation_steps) < 1 or args.learning_rate <= 0:
        parser.error("Epochs, batch size, threads, and learning rate must be positive.")
    run(args)


if __name__ == "__main__":
    main()

"""Train SarcasmLens from CSV or Excel datasets.

Run from project root:

python -m backend.train --data data/mustard_sarcasm_dataset.xlsx

The model uses:
    context + response -> sarcasm label
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import shutil
import uuid

from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from backend.model import (
    BASE_MODEL,
    DATA_PATH,
    ID2LABEL,
    LABEL2ID,
    MAX_LENGTH,
    MODEL_DIR,
    OUTPUT_DIR,
)


# ---------------------------------------------------------
# Random seed
# ---------------------------------------------------------

def seed_everything(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------
# Convert context lists to normal text
# ---------------------------------------------------------

def clean_context(value):

    if isinstance(value, list):
        return " ".join(str(x) for x in value)

    text = str(value).strip()

    if not text:
        return ""

    # Try JSON
    try:

        value = json.loads(text)

        if isinstance(value, list):
            return " ".join(str(x) for x in value)

    except Exception:
        pass

    # Try Python-list format
    try:

        value = ast.literal_eval(text)

        if isinstance(value, list):
            return " ".join(str(x) for x in value)

    except Exception:
        pass

    return text


# ---------------------------------------------------------
# Load CSV or Excel
# ---------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:

    if not path.is_file():

        raise ValueError(
            f"Dataset not found: {path}"
        )

    print(f"\nLoading dataset: {path}")

    # ----------------------------
    # Read file
    # ----------------------------

    if path.suffix.lower() == ".csv":

        df = pd.read_csv(
            path,
            keep_default_na=False
        )

    elif path.suffix.lower() in [".xlsx", ".xls"]:

        df = pd.read_excel(path)

        df = df.fillna("")

    else:

        raise ValueError(
            "Dataset must be CSV or Excel (.xlsx/.xls)"
        )

    print("Original rows:", len(df))
    print("Columns:", list(df.columns))

    # -------------------------------------------------
    # MUStARD format
    # -------------------------------------------------

    if {
        "key",
        "utterance",
        "speaker",
        "context",
        "show",
        "sarcasm",
    }.issubset(df.columns):

        print("\nDetected: MUStARD-style dataset")

        converted = pd.DataFrame()

        converted["scene"] = [
            f"mustard_{i + 1}"
            for i in range(len(df))
        ]

        converted["key"] = df["key"].astype(str)

        converted["context"] = (
            df["context"]
            .apply(clean_context)
        )

        converted["response"] = (
            df["utterance"]
            .astype(str)
            .str.strip()
        )

        converted["label"] = pd.to_numeric(
            df["sarcasm"],
            errors="coerce"
        )

        if "sarcasm_type" in df.columns:

            converted["sarcasm_type"] = (
                df["sarcasm_type"]
                .astype(str)
            )

        else:

            converted["sarcasm_type"] = "unknown"

        converted["speaker"] = (
            df["speaker"]
            .astype(str)
        )

        converted["show"] = (
            df["show"]
            .astype(str)
        )

        converted["implicit_emotion"] = "unknown"
        converted["explicit_emotion"] = "unknown"
        converted["valence"] = "unknown"
        converted["arousal"] = "unknown"

        df = converted

    # -------------------------------------------------
    # Existing SarcasmLens format
    # -------------------------------------------------

    elif {
        "scene",
        "key",
        "context",
        "response",
        "label",
    }.issubset(df.columns):

        print("\nDetected: SarcasmLens dataset")

    else:

        raise ValueError(
            "\nUnsupported dataset structure.\n"
            "MUStARD requires: "
            "key, utterance, speaker, context, show, sarcasm\n"
            "OR SarcasmLens requires: "
            "scene, key, context, response, label"
        )

    # -------------------------------------------------
    # Validate labels
    # -------------------------------------------------

    df["label"] = pd.to_numeric(
        df["label"],
        errors="coerce"
    )

    invalid_labels = df["label"].isna()

    if invalid_labels.any():

        print(
            "Removing rows with invalid labels:",
            int(invalid_labels.sum())
        )

        df = df[~invalid_labels].copy()

    df["label"] = df["label"].astype(int)

    df = df[
        df["label"].isin([0, 1])
    ].copy()

    # -------------------------------------------------
    # Clean context/response
    # -------------------------------------------------

    df["context"] = (
        df["context"]
        .astype(str)
        .str.strip()
    )

    df["response"] = (
        df["response"]
        .astype(str)
        .str.strip()
    )

    before = len(df)

    df = df[
        (df["context"] != "")
        &
        (df["response"] != "")
    ].copy()

    removed = before - len(df)

    if removed:

        print(
            f"Removed {removed} rows "
            "with empty context/response."
        )

    if len(df) < 10:

        raise ValueError(
            "Not enough valid rows for training."
        )

    # -------------------------------------------------
    # Statistics
    # -------------------------------------------------

    print("\nFinal dataset rows:", len(df))

    print("\nClass distribution:")

    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    print("\n0 = NOT SARCASM")
    print("1 = SARCASM")

    if df["label"].nunique() < 2:

        raise ValueError(
            "Dataset must contain both "
            "sarcastic and non-sarcastic examples."
        )

    return df.reset_index(drop=True)


# ---------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------

class ConversationDataset(Dataset):

    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer
    ):

        self.encodings = tokenizer(
            frame["context"].tolist(),
            frame["response"].tolist(),
            truncation="longest_first",
            max_length=MAX_LENGTH,
            padding=False,
        )

        self.labels = (
            frame["label"]
            .astype(int)
            .tolist()
        )

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        item = {
            key: value[index]
            for key, value
            in self.encodings.items()
        }

        item["labels"] = self.labels[index]

        return item


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def calculate_metrics(labels, predictions):

    return {

        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),

        "sarcasm_f1": float(
            f1_score(
                labels,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),

        "accuracy": float(
            accuracy_score(
                labels,
                predictions
            )
        ),

        "precision": float(
            precision_score(
                labels,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                labels,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),

        "confusion_matrix":
            confusion_matrix(
                labels,
                predictions,
                labels=[0, 1],
            ).tolist(),
    }


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

@torch.inference_mode()
def evaluate(
    model,
    loader,
    device
):

    model.eval()

    labels = []
    probabilities = []

    for batch in loader:

        batch_labels = (
            batch.pop("labels")
            .tolist()
        )

        labels.extend(batch_labels)

        batch = {
            key: value.to(device)
            for key, value
            in batch.items()
        }

        logits = model(
            **batch
        ).logits

        probs = torch.softmax(
            logits,
            dim=-1
        )

        probabilities.extend(
            probs.cpu().tolist()
        )

    predictions = np.argmax(
        probabilities,
        axis=1
    ).tolist()

    return (
        calculate_metrics(
            labels,
            predictions
        ),
        predictions,
        probabilities,
    )


# ---------------------------------------------------------
# JSON writer
# ---------------------------------------------------------

def write_json(path: Path, value):

    temporary = path.with_suffix(
        ".json.tmp"
    )

    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def train(args):

    seed_everything(args.seed)

    frame = load_data(args.data)

    # ----------------------------------------
    # Train / Validation split
    # ----------------------------------------

    train_df, validation_df = train_test_split(
        frame,
        test_size=0.20,
        stratify=frame["label"],
        random_state=args.seed,
    )

    print(
        f"\nTraining rows: {len(train_df)}"
    )

    print(
        f"Validation rows: {len(validation_df)}"
    )

    # ----------------------------------------
    # Check overlap
    # ----------------------------------------

    train_pairs = set(
        zip(
            train_df.context,
            train_df.response
        )
    )

    validation_pairs = set(
        zip(
            validation_df.context,
            validation_df.response
        )
    )

    scene_overlap = (
        set(train_df.scene)
        &
        set(validation_df.scene)
    )

    overlap = {

        "shared_text_pairs":
            len(
                train_pairs
                &
                validation_pairs
            ),

        "shared_scenes":
            len(scene_overlap),
    }

    # ----------------------------------------
    # Device
    # ----------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    print(
        f"Split overlap audit: {overlap}"
    )

    # ----------------------------------------
    # Tokenizer + model
    # ----------------------------------------

    print(
        f"\nLoading model: {BASE_MODEL}"
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            BASE_MODEL
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            BASE_MODEL,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
    )

    model.to(device)

    collator = (
        DataCollatorWithPadding(
            tokenizer=tokenizer
        )
    )

    generator = (
        torch.Generator()
        .manual_seed(args.seed)
    )

    # ----------------------------------------
    # DataLoaders
    # ----------------------------------------

    train_loader = DataLoader(

        ConversationDataset(
            train_df,
            tokenizer
        ),

        batch_size=args.batch_size,

        shuffle=True,

        generator=generator,

        collate_fn=collator,

        num_workers=0,
    )

    validation_loader = DataLoader(

        ConversationDataset(
            validation_df,
            tokenizer
        ),

        batch_size=args.batch_size,

        shuffle=False,

        collate_fn=collator,

        num_workers=0,
    )

    # ----------------------------------------
    # Optimizer
    # ----------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    total_steps = (
        args.epochs
        *
        len(train_loader)
    )

    scheduler = (
        get_linear_schedule_with_warmup(

            optimizer,

            num_warmup_steps=
                int(
                    0.1
                    *
                    total_steps
                ),

            num_training_steps=
                total_steps,
        )
    )

    # ----------------------------------------
    # Output directory
    # ----------------------------------------

    run_id = (
        datetime.now(
            timezone.utc
        ).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        +
        "_"
        +
        uuid.uuid4().hex[:8]
    )

    run_dir = (
        OUTPUT_DIR
        /
        run_id
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    staging = (
        run_dir
        /
        "best_checkpoint"
    )

    # ----------------------------------------
    # Save split information
    # ----------------------------------------

    train_df[
        [
            "scene",
            "key",
            "label"
        ]
    ].to_csv(
        run_dir
        /
        "train_split.csv",
        index=False
    )

    validation_df[
        [
            "scene",
            "key",
            "label"
        ]
    ].to_csv(
        run_dir
        /
        "validation_split.csv",
        index=False
    )

    history = []

    best_score = -1.0
    best_epoch = None
    best_metrics = None

    # ----------------------------------------
    # Training loop
    # ----------------------------------------

    for epoch in range(
        1,
        args.epochs + 1
    ):

        model.train()

        total_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=
                f"Epoch "
                f"{epoch}/"
                f"{args.epochs}"
        )

        for batch in progress:

            batch = {
                key: value.to(device)
                for key, value
                in batch.items()
            }

            optimizer.zero_grad(
                set_to_none=True
            )

            loss = model(
                **batch
            ).loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            scheduler.step()

            total_loss += (
                loss.item()
                *
                batch["labels"].size(0)
            )

            progress.set_postfix(
                loss=
                    f"{loss.item():.4f}"
            )

        # ------------------------------------
        # Validation
        # ------------------------------------

        metrics, predictions, probabilities = evaluate(
            model,
            validation_loader,
            device
        )

        history.append({

            "epoch": epoch,

            "training_loss":
                total_loss
                /
                len(train_df),

            **metrics,
        })

        write_json(
            run_dir
            /
            "training_history.json",
            history
        )

        print(
            f"\nValidation epoch "
            f"{epoch}:"
        )

        print(
            json.dumps(
                metrics,
                indent=2
            )
        )

        # ------------------------------------
        # Save best model
        # ------------------------------------

        if (
            metrics["macro_f1"]
            >
            best_score
        ):

            best_score = (
                metrics["macro_f1"]
            )

            best_epoch = epoch

            best_metrics = metrics

            staging.mkdir(
                parents=True,
                exist_ok=True
            )

            model.save_pretrained(
                staging,
                safe_serialization=True
            )

            tokenizer.save_pretrained(
                staging
            )

            validation_output = (
                validation_df[
                    [
                        "scene",
                        "key",
                        "label"
                    ]
                ].copy()
            )

            validation_output[
                "prediction"
            ] = predictions

            validation_output[
                "not_sarcasm_probability"
            ] = [
                p[0]
                for p
                in probabilities
            ]

            validation_output[
                "sarcasm_probability"
            ] = [
                p[1]
                for p
                in probabilities
            ]

            validation_output.to_csv(
                run_dir
                /
                "validation_predictions.csv",
                index=False
            )

    # ----------------------------------------
    # Final report
    # ----------------------------------------

    report = {

        **best_metrics,

        "training_complete": True,

        "best_epoch":
            best_epoch,

        "selection_metric":
            "macro_f1",

        "base_model":
            BASE_MODEL,

        "max_length":
            MAX_LENGTH,

        "features":
            [
                "context",
                "response"
            ],

        "seed":
            args.seed,

        "epochs":
            args.epochs,

        "learning_rate":
            args.learning_rate,

        "batch_size":
            args.batch_size,

        "train_size":
            len(train_df),

        "validation_size":
            len(validation_df),

        "dataset":
            str(args.data),

        "dataset_sha256":
            hashlib.sha256(
                args.data.read_bytes()
            ).hexdigest(),

        "device":
            str(device),

        "run_id":
            run_id,

        "completed_at":
            datetime.now(
                timezone.utc
            ).isoformat(),
    }

    write_json(
        staging
        /
        "metrics.json",
        report
    )

    write_json(
        run_dir
        /
        "metrics.json",
        report
    )

    # ----------------------------------------
    # Publish to saved_model
    # ----------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for file in staging.iterdir():

        shutil.copy2(
            file,
            MODEL_DIR
            /
            file.name
        )

    print(
        "\n================================"
    )

    print(
        "TRAINING COMPLETED SUCCESSFULLY"
    )

    print(
        "================================"
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Macro F1: {best_score:.4f}"
    )

    print(
        f"Model saved to: {MODEL_DIR}"
    )


# ---------------------------------------------------------
# Arguments
# ---------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    if (
        args.epochs < 1
        or
        args.batch_size < 1
        or
        args.learning_rate <= 0
    ):

        parser.error(
            "epochs, batch-size and "
            "learning-rate must be positive"
        )

    return args


# ---------------------------------------------------------
# Start
# ---------------------------------------------------------

if __name__ == "__main__":

    try:

        train(
            parse_args()
        )

    except (
        ValueError,
        OSError
    ) as exc:

        raise SystemExit(
            f"Training stopped: {exc}"
        ) from exc
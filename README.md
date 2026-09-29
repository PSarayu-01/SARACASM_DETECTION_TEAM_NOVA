# SarcasmLens AI

## Estimated sarcasm type

Context-aware `/predict` results and `/compare.context_aware` now include `sarcasm_type`: a status, category code/name, conditional confidence, probabilities, and a limitation note. The dashboard shows this in the result card and saved history. The Performance tab loads separate type metrics from `GET /type-metrics`. Health includes `type_model_loaded`.

Supported dataset categories are **PRO — Propositional**, **ILL — Illocutionary**, and **EMB — Embedded**, following the [MUStARD++ taxonomy](https://aclanthology.org/2022.lrec-1.756/). The supplied CSV has only three LIK (Like-prefixed) examples; that category is explicitly unsupported. A non-sarcastic binary prediction returns `not_applicable`; a missing type checkpoint returns `unavailable` without preventing binary predictions.

This is a separate experimental model: frozen, original pretrained RoBERTa-base encodes context and response as a text pair (256-token limit). A class-weighted linear softmax head is trained on the resulting mean-pooled features. `sarcasm_type` is the target only, never an input to this model or the binary classifier. No emotions, scores, or metadata enter the encoder. Feature normalization uses training rows only. The binary checkpoint is unchanged.

```powershell
.\.venv\Scripts\python.exe -m backend.train_types
```

This requires the pretrained RoBERTa-base already present in the Hugging Face cache from the original training. Artifacts go to `saved_type_model/`; audit records go to `outputs/types_<timestamp>/`. Restart the backend after training. Fresh source ZIPs exclude type weights as well as binary weights.

Initial measured results: **49.1% Macro F1 and 55.8% accuracy**, on 120 validation examples (478 training examples). The best head epoch was 26 of 100; encoder weights were frozen throughout. These are validation-selected, conditional type scores on labelled sarcastic examples, not independent test accuracy or combined binary-plus-type accuracy. Type confidence is uncalibrated and does not establish that sarcasm is present. The displayed category is an estimate among the supported categories, not a generated explanation of the speaker's intent.

Context-Aware Sarcasm & Irony Detection using a locally fine-tuned RoBERTa-base binary classifier. This project contains the supplied dataset, training pipeline, FastAPI backend, and responsive conversation dashboard.

**Fresh source ZIPs do not include trained model weights.** Run training before inference. `saved_model/metrics.json` is created only from actual validation results. There are no fabricated metrics or fallback predictions.

## Windows / VS Code quick start

Install 64-bit Python 3.11 or 3.12 and open this folder in VS Code. Run every command below in the project root using PowerShell. Commands call the virtual environment directly, so no execution-policy change or activation is required.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --timeout 120 -r requirements.txt
.\.venv\Scripts\python.exe -m backend.train
```

If you installed Python 3.11, replace `py -3.12` with `py -3.11`. The first training run downloads pretrained RoBERTa weights and its tokenizer from Hugging Face. This needs an internet connection and disk space; text is not sent to an AI inference API. Training and subsequent inference run locally. CPU training can be slow. The classifier-head initialization warning on first training is expected: the new classification head is learned from this dataset.

After training finishes, start the backend in terminal 1:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open another VS Code terminal in the same project root and serve the frontend:

```powershell
.\.venv\Scripts\python.exe -m http.server 5500 --bind 127.0.0.1 --directory frontend
```

Open **http://127.0.0.1:5500** in your browser. API docs: **http://127.0.0.1:8000/docs**. Do not open the HTML as a `file://` URL. Stop either server with Ctrl+C. Stop the API before retraining and restart it afterwards; it deliberately loads the model once.

For NVIDIA CUDA acceleration, install a compatible CUDA-enabled PyTorch build into `.venv` using the Windows/Pip command from [the official PyTorch installation selector](https://docs.pytorch.org/get-started/locally/) before installing the other requirements. Replace `pip` in its command with `.\.venv\Scripts\python.exe -m pip`. Then check:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Training and inference automatically select CUDA when available, otherwise CPU. If you run out of memory, train with a smaller batch:

```powershell
.\.venv\Scripts\python.exe -m backend.train --batch-size 2
```

## Project layout

```text
SarcasmLens_AI/
  backend/
    __init__.py
    train.py
    model.py
    app.py
  frontend/
    index.html
    style.css
    script.js
  data/mustard_context_sarcasm_cleaned.csv
  saved_model/                 # Best trained model, tokenizer, metrics.json
  outputs/                     # Per-run audit records and best checkpoint
  tests/test_backend.py
  requirements.txt
  requirements-dev.txt
  .vscode/settings.json
  .vscode/launch.json
  README.md
```

## Training and evaluation

- Base checkpoint: `FacebookAI/roberta-base`, the official RoBERTa-base repository. `AutoModelForSequenceClassification`, two labels: `0=NOT_SARCASM`, `1=SARCASM`.
- Exactly 1,202 rows and 601 examples per class are validated before training. Empty inputs, invalid labels, or incorrect schema stop training with an actionable error; rows are never silently discarded.
- Only `context` and `response` enter the tokenizer as a text pair. `label` is the target. All other columns, including `sarcasm_type`, emotions, valence, arousal, speaker and show, are excluded from model features.
- Maximum 256 tokens including special tokens; longest-first truncation and dynamic batch padding. Long input may lose information. Inference uses the same pair encoding and limit.
- Stratified 80/20 row split, `random_state=42`: 961 training and 241 validation rows. Fixed Python, NumPy and PyTorch seeds; deterministic settings where supported. Exact bitwise reproducibility across hardware/software is not guaranteed.
- Defaults: 3 epochs, AdamW learning rate `2e-5`, weight decay `0.01`, batch size 8, 10% linear warmup, linear learning-rate decay, gradient clipping at 1.0.
- Each epoch is evaluated; the highest **Macro F1** checkpoint wins. Ties keep the earlier epoch. A run saves real Macro F1, sarcasm-class F1, accuracy, and sarcasm-class precision/recall. Scores use a 0–1 scale; the dashboard formats them as percentages.
- Confusion matrix order: `[[true_negative, false_positive], [false_negative, true_positive]]`. Rows are true labels and columns predicted labels, both ordered `[NOT_SARCASM, SARCASM]`.
- `saved_model/metrics.json` includes training settings, dataset hash, split sizes, best epoch, timestamp, and overlap counts. Each `outputs/<run_id>/` preserves training history, train/validation row assignments, best-validation predictions and probabilities, metrics, and a backup best checkpoint.
- The validation set selects the best epoch; its score is **not an independent test score**. Row-level splitting can share scenes or repeated dialogue across train and validation. The run reports overlap. A scene-grouped split and untouched test set would be useful further evaluations; this project preserves the requested stratified split.

## API

`GET /health` returns HTTP 200 when the API is running, with `model_loaded`, `device`, and status `ok` or `model_missing`. Model readiness is distinct from server availability.

`GET /metrics` returns the saved metrics for the loaded checkpoint, or HTTP 503 when the trained model is unavailable.

`POST /predict` accepts:

```json
{"context":"I failed my exam.","response":"Fantastic, exactly what I wanted."}
```

The response contains `prediction`, `confidence`, `sarcasm_probability`, and `not_sarcasm_probability`. Probabilities are softmax over the actual local model logits. Confidence is the probability of the predicted class, not always sarcasm probability. No illustrative probability numbers are included here because inference has not yet run.

`POST /compare` accepts the same payload and returns `response_only` and `context_aware`, each with the prediction schema above. Response-only uses a single input sequence; context-aware uses the text pair. Both use the **same** fine-tuned checkpoint. This is an inference ablation, not a separately trained baseline or proof that context always improves a prediction. The frontend uses this endpoint for both the main result and comparison, avoiding a redundant third forward pass.

Blank or whitespace-only fields produce HTTP 422. Context is capped at 20,000 characters, response at 10,000. Unavailable models return HTTP 503, inference failures return a friendly HTTP 500. Server logs retain diagnostic details. CORS allows localhost/127.0.0.1 on ports 5500 and 3000, with GET/POST and Content-Type.

Manual API check in PowerShell after training:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
$body = @{ context = 'I failed my exam.'; response = 'Fantastic, exactly what I wanted.' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType 'application/json' -Body $body
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/compare -ContentType 'application/json' -Body $body
Invoke-RestMethod http://127.0.0.1:8000/metrics
```

## Dashboard and data handling

The dashboard includes a conversation editor, actual softmax probability bar, comparison, real validation metrics, and the latest ten analyses in browser localStorage. Examples and history reuse only populate input fields; they never create predictions. Editing inputs clears the previous result. Saved history entries remain explicitly labelled as historical results. Clear history removes the saved entries. Conversation text is rendered as text, not HTML. If browser storage fails, history remains available in memory for that page session.

No external inference APIs, keyword detection, random probabilities, remote fonts, or JavaScript frameworks are used. The API does not persist submitted conversations; the browser history does. Softmax probabilities are uncalibrated model confidence, not a correctness guarantee. This dataset trains binary sarcasm detection; the project does not claim separately validated irony-subtype detection.

## Checks

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q backend
```

The tests check schema/data constraints, stratification, metric arithmetic, local model absence, API validation/CORS, and real tensor-softmax inference plumbing using isolated test doubles. Test probabilities never enter the application or saved metrics. Full prediction quality can only be assessed after training the actual RoBERTa checkpoint.

Implementation references: [Hugging Face RoBERTa](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/roberta), [PyTorch installation](https://docs.pytorch.org/get-started/locally/).

## Additional synthetic-data training

The supplied `data/sarcasmlens_synthetic_2000_chatgpt.csv` contains 2,000 rows but only 320 unique normalized context-response pairs across 40 contexts. The augmented pipeline removes 1,680 duplicate rows, holds out eight entire parent-context groups (64 examples), and adds the remaining 256 examples to the original 961 training examples. The original 241 validation rows remain unchanged. Only full dialogue context and response are tokenized; labels and source metadata never enter the model inputs.

After the original training has completed:

```powershell
.\.venv\Scripts\python.exe -m backend.train_augmented --prepare-only
.\.venv\Scripts\python.exe -u -m backend.train_augmented --epochs 3 --threads 4
```

This continues training from your current local checkpoint at learning rate `1e-5`, using microbatches of two and four gradient-accumulation steps (effective batch size eight) to reduce memory use. It evaluates the starting model and every epoch on the original validation set and the separately reported synthetic holdout. Results, data manifests, progress, and candidate checkpoints are saved under `outputs/augmented_<timestamp>/`. `outputs/latest_augmented_run.json` identifies the latest run. The active model remains unchanged during training.

`comparison.json` identifies an accepted candidate only if original-validation Macro F1 improves and original-validation false positives do not increase. Among qualifying epochs, the highest Macro F1 wins. These remain model-selection results, not independent test results. Synthetic examples repeat a small number of contexts and writing patterns, so their holdout score does not establish real-world accuracy. No individual phrase receives a special prediction rule.

To publish an accepted candidate, stop FastAPI with Ctrl+C, substitute its actual run ID below, then restart FastAPI:

```powershell
.\.venv\Scripts\python.exe -m backend.publish_augmented augmented_YYYYMMDDTHHMMSSZ
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Publication retains the previous model under that run's `previous_saved_model/` directory. If no candidate qualifies, keep the current model and review `comparison.json`; more training data is not a guarantee of improvement.

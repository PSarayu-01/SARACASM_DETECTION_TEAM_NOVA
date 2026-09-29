"""Publish an accepted augmented checkpoint, retaining the previous model.

Stop the backend first. Run python -m backend.publish_augmented RUN_ID, then
restart FastAPI. Training itself never changes the active checkpoint.
"""
import argparse
import json
import shutil
from pathlib import Path

from backend.model import MODEL_DIR, OUTPUT_DIR


def publish(run_id):
    run_dir = (OUTPUT_DIR / run_id).resolve()
    if run_dir.parent != OUTPUT_DIR.resolve():
        raise ValueError("Run ID must name a direct child of outputs/.")
    decision_path = run_dir / "comparison.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    candidate = decision.get("accepted_candidate")
    if not candidate:
        raise ValueError("No candidate met the validation criteria; keeping the existing model.")
    if decision.get("published"):
        raise ValueError("This run was already published.")
    checkpoint = Path(candidate["checkpoint"]).resolve()
    if checkpoint.parent != run_dir:
        raise ValueError("Candidate checkpoint is outside the selected run.")
    metadata = json.loads((checkpoint / "metrics.json").read_text(encoding="utf-8"))
    current = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
    if current["run_id"] != metadata["initial_checkpoint_run_id"]:
        raise ValueError("Active model changed since this run began. Re-evaluate before publishing.")
    backup = run_dir / "previous_saved_model"
    staging = run_dir / "publish_staging"
    if backup.exists() or staging.exists():
        raise ValueError("Publication paths already exist; inspect them before retrying.")
    # Copy fully before renaming; an interrupted copy leaves the old model intact.
    shutil.copytree(checkpoint, staging)
    MODEL_DIR.rename(backup)
    try:
        staging.rename(MODEL_DIR)
    except Exception:
        backup.rename(MODEL_DIR)
        raise
    decision.update({"published": True, "previous_model_backup": str(backup)})
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(f"Published epoch {candidate['epoch']}. Previous model preserved in {backup}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    publish(parser.parse_args().run_id)

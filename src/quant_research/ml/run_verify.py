"""Verification of a published walk-forward run directory (docs/ml-plan.md, PR #13 review).

Two layers:
- Integrity: the manifest records a SHA-256 of every fold file; any change to a file is detected. This is not a
  signature: whoever rewrites the files and the manifest can recompute the hashes.
- Consistency: from the stored predictions, the test keys (unique within and across folds, equal to the recorded
  key hash), the per-fold constants, the daily incremental IC, the fold means and the primary test are recomputed
  and must equal what was stored. With the verified dataset, every prediction is also recomputed from its loaded
  model.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.ml import artifacts, evaluation, training
from quant_research.ml.canonical import digest
from quant_research.ml.contracts import ContractError

RUN_SCHEMA = "ml-run/1"
COUNT_KEYS = ("test_dates", "too_few_stocks_dates", "undefined_ic_dates", "valid_dates")


def fold_dir(root: Path, i: int) -> Path:
    return Path(root) / "folds" / f"{i:03d}"


def dates(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s).dt.date.to_numpy()


def key_hash(day: np.ndarray, symbols) -> str:
    return digest([[d, str(s)] for d, s in zip(day, symbols, strict=True)])


def eval_frame(pred: pd.DataFrame) -> pd.DataFrame:
    """The evaluation input of one fold, built the same way at run time and at verification."""
    return pd.DataFrame({"date": dates(pred["date"]), "symbol": pred["symbol"].astype(str).to_numpy(),
                         "prediction": pred["prediction"].to_numpy(dtype=np.float64),
                         "baseline": pred["baseline_score"].to_numpy(dtype=np.float64),
                         "label": pred["label"].to_numpy(dtype=np.float64)})


def fold_means(daily: pd.DataFrame) -> dict:
    return {"model_mean_ic": float(daily["model_ic"].mean()), "baseline_mean_ic": float(daily["baseline_ic"].mean()),
            "incremental_mean_ic": float(daily["incremental_ic"].mean())}


def file_hashes(root: Path) -> dict[str, str]:
    """SHA-256 of every file under folds/ (lock files excluded), keyed by its path relative to the run root."""
    root = Path(root)
    out = {}
    for p in sorted((root / "folds").rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            out[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def _json(x):
    return json.loads(json.dumps(x, default=str))


def _fail(path, why: str) -> None:
    raise ContractError(f"{path}: {why}")


def verify_run(path: Path, dataset=None) -> dict:
    """Return the manifest of a consistent run directory, or raise ContractError. `dataset`: the run's dataset (a
    DatasetResult or a read_dataset pair); when given, predictions are recomputed from the fold models."""
    root = Path(path)
    try:
        manifest = json.loads((root / "manifest.json").read_text())
    except (OSError, ValueError) as e:
        raise ContractError(f"{root}: unreadable run manifest ({type(e).__name__})") from None
    if manifest.get("schema") != RUN_SCHEMA:
        _fail(root, f"unsupported run schema {manifest.get('schema')!r}")
    if file_hashes(root) != manifest.get("files"):
        _fail(root, "the fold files do not match the hashes in the manifest")
    if dataset is not None and training.open_dataset(dataset).fingerprint != manifest["dataset_fingerprint"]:
        _fail(root, "the dataset given is not the run's dataset")
    horizon = manifest["experiment"]["dataset"]["label"]["horizon"]
    seen, series = set(), []
    for m in manifest["folds"]:
        out, i = fold_dir(root, m["fold"]), m["fold"]
        trained = artifacts.load_model(out / "model")
        if trained.model_id != m["model_id"]:
            _fail(root, f"fold {i} model does not match its metrics (model_id)")
        pred = pd.read_parquet(out / "predictions.parquet")
        if len(pred) != m["test_rows"]:
            _fail(root, f"fold {i} predictions are incomplete")
        consts = {"fold": i, "model_id": m["model_id"], "baseline_sign": m["baseline_sign"],
                  "dataset_fingerprint": manifest["dataset_fingerprint"]}
        if any(set(pred[c]) != {v} for c, v in consts.items()):
            _fail(root, f"fold {i} predictions carry another fold, model, sign or dataset")
        day, sym = dates(pred["date"]), pred["symbol"].astype(str).to_numpy()
        keys = set(zip(day, sym, strict=True))
        if len(keys) != len(pred) or keys & seen:
            _fail(root, f"fold {i} has duplicate test keys, within the fold or across folds")
        seen |= keys
        if key_hash(day, sym) != m["test_keys_hash"]:
            _fail(root, f"fold {i} test keys differ from the recorded key hash")
        daily, counts = evaluation.daily_incremental_ic(eval_frame(pred), m["baseline_sign"])
        stored = pd.read_parquet(out / "daily_ic.parquet").assign(date=lambda x: dates(x["date"]))
        if not (stored.columns.equals(daily.columns) and stored.reset_index(drop=True).equals(daily)):
            _fail(root, f"fold {i} daily IC differs from the IC recomputed from its predictions")
        if any(counts[k] != m[k] for k in COUNT_KEYS) or any(v != m[k] for k, v in fold_means(daily).items()):
            _fail(root, f"fold {i} metrics differ from the values recomputed from its predictions")
        if dataset is not None:
            again = training.predict_model(trained, dataset, pred["row"].to_numpy())
            if not (np.array_equal(again["prediction"].to_numpy(), pred["prediction"].to_numpy())
                    and np.array_equal(again["label"].to_numpy(), pred["label"].to_numpy())):
                _fail(root, f"fold {i} predictions or labels differ from the model and dataset")
        series.append(daily)
    primary = evaluation.primary_test(pd.concat(series, ignore_index=True), horizon)
    if _json(primary) != manifest.get("primary"):
        _fail(root, "the primary test differs from the one recomputed from the daily IC")
    return manifest

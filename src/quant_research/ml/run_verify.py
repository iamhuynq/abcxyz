"""Verification of a published walk-forward run directory (docs/ml-plan.md, PR #13 reviews).

What is checked, by level:
1. `verify_run(path)`: the directory is internally consistent.
   - Integrity: the manifest records a SHA-256 of every fold file, and each is recomputed.
   - Consistency: from the stored predictions, the test keys (unique within and across folds, equal to the
     recorded key hash), the per-fold constants, the daily incremental IC, the fold metrics (each fold's
     metrics.json equals its manifest entry) and the primary test are recomputed and must equal what was stored.
2. `verify_run(path, dataset, baseline)` (or runner.reverify, which rebuilds both from research.duckdb): the
   stored rows, dates, symbols, labels and baseline scores equal the dataset rows and the baseline of the
   recorded factor build, and every prediction is recomputed from its fold model. The baseline can be re-verified
   only while research.duckdb holds the run's build.
3. Neither level proves who wrote the directory: a complete, consistent replacement of the directory passes both.
   SHA-256 here is an integrity check, not a signature. The authoritative record is research_runs and
   hypothesis_log (status and the logged p-value); a consumer compares the manifest with it.
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


def verify_run(path: Path, dataset=None, baseline: np.ndarray | None = None) -> dict:
    """Return the manifest of a consistent run directory, or raise ContractError (levels in the module docstring).
    `dataset`: the run's dataset (a DatasetResult or a read_dataset pair); when given, predictions are recomputed
    from the fold models and the stored keys and labels compared with the dataset rows. `baseline`: the baseline
    score of every dataset row (runner.load_baseline), compared with the stored baseline_score; needs `dataset`."""
    if baseline is not None and dataset is None:
        raise ContractError("a baseline can be verified only together with its dataset")
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
        try:
            stored_metrics = json.loads((out / "metrics.json").read_text())
        except (OSError, ValueError):
            stored_metrics = None
        if stored_metrics != m:
            _fail(root, f"fold {i} metrics.json differs from the manifest")
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
            rows = pred["row"].to_numpy()
            again = training.predict_model(trained, dataset, rows)               # keys and labels: the dataset's
            # row identity: the dataset row named by `row` must carry the stored date, symbol, label and prediction
            same = (np.array_equal(dates(again["date"]), day)
                    and np.array_equal(again["symbol"].astype(str).to_numpy(), sym)
                    and np.array_equal(again["label"].to_numpy(), pred["label"].to_numpy())
                    and np.array_equal(again["prediction"].to_numpy(), pred["prediction"].to_numpy()))
            if not same:
                _fail(root, f"fold {i} rows, keys, labels or predictions differ from the model and dataset")
            if baseline is not None and not np.array_equal(np.asarray(baseline, dtype=np.float64)[rows],
                                                          pred["baseline_score"].to_numpy(dtype=np.float64),
                                                          equal_nan=True):
                _fail(root, f"fold {i} baseline scores differ from the factor build")
        series.append(daily)
    primary = evaluation.primary_test(pd.concat(series, ignore_index=True), horizon)
    if _json(primary) != manifest.get("primary"):
        _fail(root, "the primary test differs from the one recomputed from the daily IC")
    return manifest

"""Walk-forward runner: one ExperimentSpec, one verified research-period dataset, one logged primary test
(docs/ml-plan.md, PR #13).

1. Read: the dataset (built from research.duckdb, read-only), the market sessions of the dataset period, and the
   baseline score of every dataset row. Nothing after the research cutoff is queried.
2. Preflight, before any run or hypothesis row: folds from ml.splits; per fold the training rows (training window,
   purged by actual label intervals) and the test rows; check_fold on those exact frames; the stock and date
   thresholds on test keys; the baseline sign from the purged training rows. Test labels are not read.
3. Idempotency: one research result per (experiment_id, dataset fingerprint); see `previous_runs`.
4. Run: start_run, primary label logged with a NULL p-value, every fold fitted on its training rows and predicted
   on its test rows (labels joined after prediction), the run directory published atomically, then the p-value
   and finish_run in one transaction. research_runs is the only source of truth for the run's status.
"""

import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research.ml import artifacts, evaluation, publish, training
from quant_research.ml.builds import check_build
from quant_research.ml.canonical import normalize
from quant_research.ml.contracts import (
    RESEARCH_END,
    ContractError,
    ExperimentSpec,
    ModelSpec,
    SplitSpec,
)
from quant_research.ml.dataset import build_dataset
from quant_research.ml.run_verify import (
    RUN_SCHEMA,
    eval_frame,
    file_hashes,
    fold_dir,
    fold_means,
    key_hash,
    verify_run,
)
from quant_research.ml.splits import Fold, check_fold, folds, purge
from quant_research.results import ResearchParams, ResultsStore

INTERVAL = ["date", "label_start", "label_end"]
__all__ = ["RUN_SCHEMA", "ExperimentRefused", "preflight", "reverify", "run_experiment", "verify_run"]


class ExperimentRefused(ContractError):
    pass


@dataclass(frozen=True)
class FoldPlan:
    fold: Fold
    train_rows: np.ndarray           # dataset row positions, strictly increasing
    test_rows: np.ndarray
    candidate_train_rows: int
    sign: int
    train_mean_ic: float
    train_valid_dates: int
    test_eligible_dates: int


def _dates(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s).dt.date.to_numpy()


def load_calendar(con, spec: ExperimentSpec, schema: str = "rs.") -> list[date]:
    """Market sessions in the dataset period (never past the research cutoff)."""
    start, end = spec.dataset.period_start, spec.dataset.period_end
    if end > RESEARCH_END:
        raise ExperimentRefused("the dataset period ends after the research cutoff")
    rows = con.execute(f"SELECT DISTINCT date FROM {schema}market_daily WHERE date BETWEEN ? AND ? ORDER BY date",
                       [start, end]).fetchall()
    return [r[0] for r in rows]


def load_baseline(con, spec: ExperimentSpec, data: pd.DataFrame, schema: str = "rs.") -> np.ndarray:
    """The BaselineSpec score of every dataset row, aligned by (date, symbol); NaN where the build has none."""
    b = spec.baseline                                 # field validated against allowlist.FACTOR_FIELDS
    scores = con.execute(f"""SELECT date, symbol, {b.field} AS baseline FROM {schema}stock_factors
                             WHERE factor = ? AND date BETWEEN ? AND ?""",
                         [b.factor, spec.dataset.period_start, min(spec.dataset.period_end, RESEARCH_END)]).df()
    if scores.duplicated(["date", "symbol"]).any():
        raise ExperimentRefused(f"stock_factors has duplicate ({b.factor}, date, symbol) rows")
    scores["date"] = _dates(scores["date"])
    keys = pd.DataFrame({"date": _dates(data["date"]), "symbol": data["symbol"].astype(str).to_numpy()})
    merged = keys.merge(scores.astype({"symbol": str}), on=["date", "symbol"], how="left", validate="one_to_one")
    return merged["baseline"].to_numpy(dtype=np.float64)


def preflight(ds: training.Dataset, baseline: np.ndarray, calendar: list[date], split: SplitSpec) -> list[FoldPlan]:
    """Every fold, checked, before anything is fitted or logged. Raises ExperimentRefused."""
    if split.step_sessions < split.test_sessions:
        raise ExperimentRefused("test windows overlap (step_sessions < test_sessions): a row would be predicted twice")
    if split.test_sessions < evaluation.MIN_DATES:
        raise ExperimentRefused(f"test_sessions {split.test_sessions} < {evaluation.MIN_DATES} valid dates per fold")
    if len(baseline) != len(ds.data):
        raise ExperimentRefused("the baseline is not aligned with the dataset")
    data, d, sym = ds.data, _dates(ds.data["date"]), ds.data["symbol"].astype(str).to_numpy()
    plans, seen = [], np.zeros(len(data), dtype=bool)
    for f in folds(calendar, split):
        cand = np.flatnonzero((d >= f.train_start) & (d <= f.train_end))
        keep = purge(data.iloc[cand][INTERVAL].reset_index(drop=True), f).to_numpy()
        train_rows, test_rows = cand[keep], np.flatnonzero((d >= f.test_start) & (d <= f.test_end))
        try:
            check_fold(data.iloc[train_rows][INTERVAL], data.iloc[test_rows][INTERVAL], f, split)
        except ContractError as e:
            raise ExperimentRefused(f"fold {f.index}: {e}") from None
        if seen[test_rows].any():
            raise ExperimentRefused(f"fold {f.index}: a test row was already tested by an earlier fold")
        seen[test_rows] = True
        test_keys = pd.DataFrame({"date": d[test_rows], "symbol": sym[test_rows], "baseline": baseline[test_rows]})
        eligible_dates = evaluation.eligible(test_keys)["date"].nunique()
        if eligible_dates < evaluation.MIN_DATES:
            raise ExperimentRefused(f"fold {f.index}: {eligible_dates} test dates with >= {evaluation.MIN_STOCKS} "
                                    f"stocks with a baseline, fewer than {evaluation.MIN_DATES}")
        train = pd.DataFrame({"date": d[train_rows], "symbol": sym[train_rows], "baseline": baseline[train_rows],
                              "label": data[ds.label].to_numpy(dtype=np.float64)[train_rows]})
        try:
            sign, mean_ic, n_dates = evaluation.baseline_sign(train)
        except ContractError as e:
            raise ExperimentRefused(f"fold {f.index}: {e}") from None
        plans.append(FoldPlan(f, train_rows, test_rows, len(cand), sign, mean_ic, n_dates, int(eligible_dates)))
    return plans


def previous_runs(store: ResultsStore, experiment_id: str, fingerprint: str, label: str, runs_dir: Path) -> list:
    """Earlier runs of the pair that block a new one: ok, running, a logged p-value, or a published directory.
    A run that failed before any result (NULL p-value, no directory) does not block a retry; an invalidated run
    never does."""
    rows = store.con.execute("""
        SELECT r.run_id, r.status, h.p_value FROM research_runs r
        LEFT JOIN hypothesis_log h ON h.run_id = r.run_id AND h.label = ?
        WHERE r.kind = 'ml' AND r.pattern_name = ? AND r.pattern_version = ? AND r.status <> 'invalid'
        ORDER BY r.created_at""", [label, experiment_id, fingerprint]).fetchall()
    return [(rid, st, p) for rid, st, p in rows
            if st in ("ok", "running") or p is not None or (runs_dir / rid).exists()]


def _key_hash(data: pd.DataFrame, rows: np.ndarray) -> str:
    sel = data.iloc[rows]
    return key_hash(_dates(sel["date"]), sel["symbol"].astype(str).to_numpy())


def _run_folds(tmp: Path, result, plans: list[FoldPlan], baseline: np.ndarray, model: ModelSpec,
               horizon: int) -> tuple[list[dict], pd.DataFrame]:
    metrics, series = [], []
    for p in plans:
        trained = training.fit_model(result, p.train_rows, model)              # training rows only
        pred = training.predict_model(trained, result, p.test_rows)             # labels attached after predicting
        pred.insert(0, "fold", p.fold.index)
        pred["baseline_score"] = baseline[p.test_rows]
        pred["baseline_sign"] = p.sign
        daily, counts = evaluation.daily_incremental_ic(eval_frame(pred), p.sign)
        if counts["valid_dates"] < evaluation.MIN_DATES:
            raise ContractError(f"fold {p.fold.index}: {counts['valid_dates']} valid dates after prediction, "
                                f"fewer than {evaluation.MIN_DATES}")
        out = fold_dir(tmp, p.fold.index)
        out.mkdir(parents=True)
        artifacts.save_model(trained, out / "model")
        pred.to_parquet(out / "predictions.parquet", index=False)
        daily.to_parquet(out / "daily_ic.parquet", index=False)
        f = p.fold
        m = {"fold": f.index, "train_start": f.train_start, "train_end": f.train_end, "test_start": f.test_start,
             "test_end": f.test_end, "embargo_start": f.embargo_start,
             "candidate_train_rows": p.candidate_train_rows, "train_rows": len(p.train_rows),
             "purged_rows": p.candidate_train_rows - len(p.train_rows), "test_rows": len(p.test_rows),
             "train_keys_hash": _key_hash(result.data, p.train_rows),
             "test_keys_hash": _key_hash(result.data, p.test_rows),
             "baseline_sign": p.sign, "baseline_train_mean_ic": p.train_mean_ic,
             "baseline_train_valid_dates": p.train_valid_dates, "test_eligible_dates": p.test_eligible_dates,
             **counts, "model_id": trained.model_id, **fold_means(daily),
             "artifact_schema": artifacts.SCHEMA_VERSION}
        (out / "metrics.json").write_text(json.dumps(m, default=str, sort_keys=True, indent=1))
        metrics.append(m)
        series.append(daily)
    return metrics, pd.concat(series, ignore_index=True)


def _lock_dir(store: ResultsStore, runs_dir: Path) -> Path:
    """`ml-locks` next to the results database (one lock namespace per results database)."""
    row = store.con.execute("SELECT path FROM duckdb_databases() WHERE database_name = current_database()").fetchone()
    return (Path(row[0]).parent if row and row[0] else Path(runs_dir)) / "ml-locks"


def _inputs(store: ResultsStore, spec: ExperimentSpec):
    """(model spec, build id, dataset result, verified dataset, aligned baseline), built the same way for a run
    and for its re-verification."""
    if not isinstance(spec, ExperimentSpec):
        raise ExperimentRefused("spec must be an ExperimentSpec")
    model = ModelSpec(spec.model, dict(spec.params), spec.seed)
    build_id = store.feature_build_id()
    if build_id is None:
        raise ExperimentRefused("research.duckdb has no feature build")
    check_build(store.con, build_id, spec.dataset, spec.baseline, schema="rs.")
    result = build_dataset(store.con, spec.dataset, build_id, schema="rs.")
    ds = training.open_dataset(result)
    return model, build_id, result, ds, load_baseline(store.con, spec, ds.data)


def reverify(store: ResultsStore, spec: ExperimentSpec, path: Path) -> dict:
    """Level-2 verification of a run directory (run_verify docstring): rebuild the dataset and the baseline from
    research.duckdb exactly as the run did, refuse unless the experiment, the dataset fingerprint and the build id
    are the manifest's, then check every stored row, key, label, baseline score and prediction. A research.duckdb
    rebuilt since the run (another build id) cannot re-verify it."""
    try:
        manifest = json.loads((Path(path) / "manifest.json").read_text())
    except (OSError, ValueError) as e:
        raise ContractError(f"{path}: unreadable run manifest ({type(e).__name__})") from None
    _model, build_id, result, ds, baseline = _inputs(store, spec)
    ours = (spec.experiment_id, ds.fingerprint, build_id)
    theirs = (manifest.get("experiment_id"), manifest.get("dataset_fingerprint"), manifest.get("build_id"))
    if ours != theirs:
        raise ContractError(f"{path}: the run was made by (experiment, dataset, build) {theirs}, not {ours}")
    return verify_run(path, result, baseline)


def run_experiment(store: ResultsStore, spec: ExperimentSpec, runs_dir: Path, now: datetime) -> tuple[str, dict]:
    """Run one experiment end to end; returns (run_id, primary test). Raises ExperimentRefused before any run
    exists when the data, the folds or an earlier run do not allow it."""
    model, build_id, result, ds, baseline = _inputs(store, spec)
    plans = preflight(ds, baseline, load_calendar(store.con, spec), spec.split)
    lock_dir = _lock_dir(store, Path(runs_dir))
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_dir / f"{spec.experiment_id}-{ds.fingerprint}.lock"
    try:                                  # one runner per (experiment, dataset): held until finish_run
        fd = publish.acquire(lock, lock)
    except FileExistsError:
        raise ExperimentRefused(f"experiment {spec.experiment_id} on dataset {ds.fingerprint} is being run by "
                                "another process; nothing was logged") from None
    try:
        return _run_locked(store, spec, model, build_id, result, ds, baseline, plans, Path(runs_dir), now)
    finally:
        os.close(fd)


def _run_locked(store, spec, model, build_id, result, ds, baseline, plans, runs_dir: Path, now) -> tuple[str, dict]:
    label, horizon = evaluation.primary_label(ds.label), spec.dataset.label.horizon
    blocking = previous_runs(store, spec.experiment_id, ds.fingerprint, label, Path(runs_dir))
    if blocking:
        raise ExperimentRefused(f"experiment {spec.experiment_id} on dataset {ds.fingerprint} already has run(s) "
                                f"{blocking}: one result per experiment; invalidate with a reason only for a defect")

    run_id = f"{now:%Y%m%dT%H%M%S}-ml-{spec.experiment_id}-research"
    config = {"experiment": normalize(spec), "experiment_id": spec.experiment_id, "model_spec": normalize(model),
              "dataset_fingerprint": ds.fingerprint, "split_version": spec.split.version, "label": label,
              "baseline_policy": evaluation.BASELINE_POLICY, "min_stocks": evaluation.MIN_STOCKS,
              "min_dates": evaluation.MIN_DATES, "run_schema": RUN_SCHEMA, "model_code_hash": training.model_code_hash(),
              "metric_definitions": evaluation.METRIC_DEFINITIONS}
    store.start_run(run_id, "ml", "research", ResearchParams(), now, (spec.experiment_id, ds.fingerprint), config)
    store.log_hypothesis(run_id, label, "research", None, now)                  # declared before any result

    out: dict = {}

    def write(tmp: Path) -> None:
        started = time.perf_counter()
        metrics, series = _run_folds(tmp, result, plans, baseline, model, horizon)
        out["primary"] = evaluation.primary_test(series, horizon)
        manifest = {"schema": RUN_SCHEMA, "run_id": run_id, **config, "build_id": build_id,
                    "dataset_metadata": json.loads(json.dumps(dict(result.metadata), default=str)),
                    "folds": metrics, "primary": out["primary"],
                    "dataset_verifications": 1 + 2 * len(plans),
                    "fold_seconds": round(time.perf_counter() - started, 3), "files": file_hashes(tmp)}
        (tmp / "manifest.json").write_text(json.dumps(manifest, default=str, sort_keys=True, indent=1))

    target = Path(runs_dir) / run_id
    with publish.locked(target):
        publish.publish_dir(target, write, verify_run, "an ML run")
    con = store.con
    con.execute("BEGIN TRANSACTION")
    try:
        store.log_hypothesis(run_id, label, "research", out["primary"]["p"], now)
        store.finish_run(run_id)
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    return run_id, out["primary"]

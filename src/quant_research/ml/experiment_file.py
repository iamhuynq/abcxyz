"""An experiment as a JSON file (docs/ml-plan.md, PR #13). Every field is explicit: nothing is defaulted, unknown or
missing keys are refused, and the contracts validate the values (source versions must be the current approved
ones, dates within research).

{"hypothesis": "...",
 "dataset": {"features": [{"table": "...", "column": "...", "source_version": "..."}],
             "label": {"column": "fwd_ret_exec_5d", "horizon": 5},
             "period_start": "YYYY-MM-DD", "period_end": "YYYY-MM-DD", "universe": "point_in_time_default"},
 "split": {"kind": "expanding", "train_min_sessions": 250, "test_sessions": 60, "step_sessions": 60,
           "embargo_sessions": 5, "first_date": "...", "last_date": "...", "train_sessions": null,
           "min_train_rows": 1000},
 "model": "ridge", "params": {"alpha": 1.0}, "seed": 0,
 "baseline": {"factor": "...", "source_version": "...", "field": "rank_pct"},
 "primary_test": "incremental_rank_ic_newey_west"}
"""

import json
from datetime import date
from pathlib import Path

from quant_research.ml.contracts import (
    BaselineSpec,
    ContractError,
    DatasetSpec,
    ExperimentSpec,
    FeatureSpec,
    LabelSpec,
    SplitSpec,
)

TOP = {"hypothesis", "dataset", "split", "model", "params", "seed", "baseline", "primary_test"}
DATASET = {"features", "label", "period_start", "period_end", "universe"}
SPLIT = {"kind", "train_min_sessions", "test_sessions", "step_sessions", "embargo_sessions", "first_date",
         "last_date", "train_sessions", "min_train_rows"}


def _keys(d, expected: set, where: str) -> dict:
    if not isinstance(d, dict) or set(d) != expected:
        got = sorted(d) if isinstance(d, dict) else type(d).__name__
        raise ContractError(f"{where} must have exactly the keys {sorted(expected)}, not {got}")
    return d


def _date(v, where: str) -> date:
    try:
        return date.fromisoformat(v)
    except (TypeError, ValueError):
        raise ContractError(f"{where} must be an ISO date, not {v!r}") from None


def parse(doc) -> ExperimentSpec:
    d = _keys(doc, TOP, "experiment")
    ds = _keys(d["dataset"], DATASET, "dataset")
    feats = ds["features"]
    if not isinstance(feats, list):
        raise ContractError("dataset.features must be a list")
    features = tuple(FeatureSpec(**_keys(f, {"table", "column", "source_version"}, "feature")) for f in feats)
    label = LabelSpec(**_keys(ds["label"], {"column", "horizon"}, "dataset.label"))
    dataset = DatasetSpec(features, label, _date(ds["period_start"], "period_start"),
                          _date(ds["period_end"], "period_end"), ds["universe"])
    sp = _keys(d["split"], SPLIT, "split")
    split = SplitSpec(**{**sp, "first_date": _date(sp["first_date"], "split.first_date"),
                         "last_date": _date(sp["last_date"], "split.last_date")})
    baseline = BaselineSpec(**_keys(d["baseline"], {"factor", "source_version", "field"}, "baseline"))
    return ExperimentSpec(d["hypothesis"], dataset, split, d["model"], d["seed"], baseline, d["primary_test"],
                          d["params"])


def load(path: Path) -> ExperimentSpec:
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise ContractError(f"{path}: unreadable experiment file ({type(e).__name__})") from None
    return parse(doc)

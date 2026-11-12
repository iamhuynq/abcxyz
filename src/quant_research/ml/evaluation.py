"""The primary test of an ML experiment: incremental daily rank IC over the signed Factor Engine baseline
(docs/ml-plan.md, PR #13). Pure functions over frames; nothing here reads a database.

- Rank IC on one date: Pearson correlation of average ranks (ties share their mean rank), as
  interactions.daily_ic. Undefined (NaN) when either side is constant.
- Samples are finite paired samples: a row counts only when every value its IC uses is finite (baseline and
  label for the baseline sign; prediction, baseline and label for the test). A date counts when at least
  MIN_STOCKS such rows remain and both ICs are defined; model and baseline always use exactly the same rows.
  Frames carry (date, symbol) keys, and duplicate keys are refused: a duplicate would weigh a stock twice.
- Baseline sign: the mean, over valid training dates (each date weighs the same), of the baseline's daily IC on
  the fold's purged training rows. > 0 keeps the score, < 0 negates it; 0 or fewer than MIN_DATES valid dates is
  an error. Test labels are never used.
- Primary statistic: the mean of the chronological daily incremental IC (model IC minus signed baseline IC),
  Newey-West with lag horizon - 1 (stats.mean_test).
"""

import math

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from quant_research import stats
from quant_research.ml.contracts import ContractError

MIN_STOCKS = 30          # rows with a finite baseline on a date
MIN_DATES = 60           # valid dates per fold (test window) and for the baseline sign (training rows)
PRIMARY_TEST = "incremental_rank_ic_newey_west"
BASELINE_POLICY = "sign of the mean daily rank IC on the fold's purged training rows; 0 or < 60 dates fails"
METRIC_DEFINITIONS = {
    "test_eligible_dates": "preflight, before prediction: test dates with >= 30 stocks with a finite baseline",
    "valid_dates": "after prediction: dates with >= 30 rows whose prediction, baseline and label are all finite and "
                   "where both rank ICs are defined; the dates of the primary test",
    "too_few_stocks_dates": "after prediction: test dates with < 30 finite paired rows",
    "undefined_ic_dates": "after prediction: dates with enough rows where either IC is undefined (constant side)",
    "baseline_train_valid_dates": "training dates with >= 30 finite (baseline, label) pairs and a defined IC"}


def primary_label(label_column: str) -> str:
    return f"ml_incremental_rank_ic_{label_column}"


def rank_ic(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation as Pearson on average ranks; NaN when either side is constant or has < 2 values."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) != len(y):
        raise ContractError("rank_ic needs two vectors of the same length")
    if len(x) < 2 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return math.nan
    rx, ry = rankdata(x) - (len(x) + 1) / 2, rankdata(y) - (len(y) + 1) / 2
    den = math.sqrt(float(rx @ rx) * float(ry @ ry))
    return float(rx @ ry) / den if den > 0 else math.nan


def check_keys(frame: pd.DataFrame) -> None:
    if frame.duplicated(["date", "symbol"]).any():
        raise ContractError("duplicate (date, symbol) rows: a stock would weigh twice in a daily rank IC")


def eligible(frame: pd.DataFrame, columns: tuple[str, ...] = ("baseline",)) -> pd.DataFrame:
    """Rows whose `columns` are all finite, on dates that keep at least MIN_STOCKS of them. The preflight passes
    ("baseline",) only: no label and no prediction exist for it."""
    check_keys(frame)
    finite = np.isfinite(frame[list(columns)].to_numpy(dtype=np.float64)).all(axis=1)
    ok = frame[finite]
    counts = ok.groupby("date")["date"].transform("size")
    return ok[counts >= MIN_STOCKS]


def daily_baseline_ic(train: pd.DataFrame) -> pd.Series:
    """Per-date IC of the raw baseline against the label on training rows (date, symbol, baseline, label)."""
    rows = eligible(train, ("baseline", "label"))
    ic = {d: rank_ic(g["baseline"].to_numpy(), g["label"].to_numpy()) for d, g in rows.groupby("date", sort=True)}
    return pd.Series(ic, dtype=np.float64).dropna()


def baseline_sign(train: pd.DataFrame) -> tuple[int, float, int]:
    """(sign, raw mean IC, valid training dates) from the fold's purged training rows only."""
    ic = daily_baseline_ic(train)
    if len(ic) < MIN_DATES:
        raise ContractError(f"baseline sign: {len(ic)} valid training dates, fewer than {MIN_DATES}")
    mean = float(ic.mean())
    if not math.isfinite(mean) or mean == 0.0:
        raise ContractError(f"baseline sign: the mean training IC is {mean}; no sign is chosen by default")
    return (1 if mean > 0 else -1), mean, len(ic)


def daily_incremental_ic(frame: pd.DataFrame, sign: int) -> tuple[pd.DataFrame, dict]:
    """frame: date, symbol, prediction, baseline, label (test rows, predictions already made). Returns the per-date
    (date, n, model_ic, baseline_ic, incremental_ic) and the counts of dropped dates (METRIC_DEFINITIONS)."""
    if sign not in (1, -1):
        raise ContractError("sign must be 1 or -1")
    all_dates = frame["date"].nunique()
    rows = eligible(frame, ("prediction", "baseline", "label"))
    out, undefined = [], 0
    for d, g in rows.groupby("date", sort=True):
        m = rank_ic(g["prediction"].to_numpy(), g["label"].to_numpy())
        b = rank_ic(sign * g["baseline"].to_numpy(dtype=np.float64), g["label"].to_numpy())
        if math.isnan(m) or math.isnan(b):
            undefined += 1                                  # dropped for both, never for one side only
            continue
        out.append((d, len(g), m, b, m - b))
    daily = pd.DataFrame(out, columns=["date", "n", "model_ic", "baseline_ic", "incremental_ic"])
    counts = {"test_dates": int(all_dates), "too_few_stocks_dates": int(all_dates - rows["date"].nunique()),
              "undefined_ic_dates": undefined, "valid_dates": len(daily)}
    return daily, counts


def primary_test(daily: pd.DataFrame, horizon: int) -> dict:
    """Mean incremental IC over the chronological dates, Newey-West lag horizon - 1. Raises when undefined."""
    if daily["date"].duplicated().any() or not daily["date"].is_monotonic_increasing:
        raise ContractError("the incremental IC series must have unique dates in chronological order")
    x = daily["incremental_ic"].to_numpy(dtype=np.float64)
    se, t, p, lo, hi = stats.mean_test(x, horizon - 1)
    if p is None or not math.isfinite(p):
        raise ContractError(f"the primary test is undefined ({len(x)} dates)")
    return {"test": PRIMARY_TEST, "n_dates": len(x), "horizon": horizon, "lag": horizon - 1,
            "mean": float(x.mean()), "se": se, "t": t, "p": p, "ci": [lo, hi]}

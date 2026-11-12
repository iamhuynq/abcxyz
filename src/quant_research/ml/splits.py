"""Chronological walk-forward folds, interval purge with an embargo gap, and the fold check every later PR calls
(docs/ml-plan.md sections 3.6, 4 and 6.1). Pure functions; dates are datetime.date.

Purge works on actual label intervals, never on row counts: a training row is kept only if its label ends
before the embargo gap that precedes the test window. The gap is `embargo_sessions` market sessions before the
test start (0: the test start itself). Training rows always precede the test window (walk-forward), so no
embargo after the test is needed.
"""

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date

import pandas as pd

from quant_research.ml.contracts import RESEARCH_END, ContractError, SplitSpec


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    embargo_start: date            # training labels must end before this session


def _calendar(calendar) -> list[date]:
    cal = sorted({pd.Timestamp(d).date() for d in calendar})
    if not cal:
        raise ContractError("empty calendar")
    return cal


def folds(calendar, split: SplitSpec) -> list[Fold]:
    """Deterministic folds over the market sessions in [split.first_date, split.last_date] (the dataset period,
    never past the research cutoff). Only complete test windows are produced; too few sessions for one fold is an
    error."""
    cal = [d for d in _calendar(calendar) if split.first_date <= d <= split.last_date]
    out, k = [], 0
    while True:
        start = split.train_min_sessions + k * split.step_sessions
        end = start + split.test_sessions - 1
        if end >= len(cal):
            break
        first = 0 if split.kind == "expanding" else max(0, start - split.train_sessions)
        gap = max(first, start - split.embargo_sessions)
        if gap <= first:
            raise ContractError(f"fold {k}: the embargo of {split.embargo_sessions} sessions leaves no training data")
        out.append(Fold(k, cal[first], cal[start - 1], cal[start], cal[end], cal[gap]))
        k += 1
    if not out:
        raise ContractError(f"{len(cal)} sessions up to {split.last_date} are too few for one fold of "
                            f"{split.train_min_sessions} + {split.test_sessions}")
    if out[-1].test_end > RESEARCH_END:                                  # unreachable through SplitSpec; kept as a guard
        raise ContractError("a fold ends after the research cutoff")
    return out


def _intervals(df: pd.DataFrame, name: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(date, label_start, label_end) as dates. Every row needs a label interval that starts after its decision
    date: the dataset builder supplies it per symbol (the symbol's own sessions) and excludes NULL labels."""
    d = pd.to_datetime(df["date"]).dt.date
    start, end = pd.to_datetime(df["label_start"]), pd.to_datetime(df["label_end"])
    if start.isna().any() or end.isna().any():
        raise ContractError(f"{name}: {int((start.isna() | end.isna()).sum())} rows without a label interval")
    start, end = start.dt.date, end.dt.date
    if ((start <= d) | (end < start)).any():
        raise ContractError(f"{name}: a label interval does not start after its decision date")
    return d, start, end


def purge(train: pd.DataFrame, fold: Fold) -> pd.Series:
    """Rows of `train` (columns date, label_start, label_end) to keep: decided inside the training window, with a
    label that ends **strictly before** fold.embargo_start (the session `embargo_sessions` before the test start).
    A label ending on embargo_start, inside the gap, or in the test window is purged; an invalid interval raises."""
    d, _start, end = _intervals(train, "train")
    return (d >= fold.train_start) & (d <= fold.train_end) & (end < fold.embargo_start)


def check_fold(train: pd.DataFrame, test: pd.DataFrame, fold: Fold, split: SplitSpec) -> None:
    """Raise unless the fold is leakage-safe. Both frames have columns date, label_start, label_end."""
    for name, df in (("train", train), ("test", test)):
        if pd.to_datetime(df["date"]).isna().any():
            raise ContractError(f"{name}: rows without date")
        for col, values in zip(("date", "label_start", "label_end"), _intervals(df, name)):
            if (values > RESEARCH_END).any():
                raise ContractError(f"{name}: {col} after the research cutoff {RESEARCH_END}")
    if fold.test_end > RESEARCH_END:
        raise ContractError(f"fold {fold.index} ends after the research cutoff {RESEARCH_END}")
    td = pd.to_datetime(train["date"]).dt.date
    if (td >= fold.test_start).any() or (td < fold.train_start).any():
        raise ContractError(f"fold {fold.index}: training rows outside the training window")
    sd = pd.to_datetime(test["date"]).dt.date
    if (sd < fold.test_start).any() or (sd > fold.test_end).any():
        raise ContractError(f"fold {fold.index}: test rows outside the test window")
    if (pd.to_datetime(train["label_end"]).dt.date >= fold.embargo_start).any():
        raise ContractError(f"fold {fold.index}: a training label reaches the embargo gap or the test window")
    if len(train) < split.min_train_rows:
        raise ContractError(f"fold {fold.index}: {len(train)} training rows < {split.min_train_rows}")
    if len(test) == 0:
        raise ContractError(f"fold {fold.index}: no test rows")


def session_offset(calendar, d: date, n: int) -> date | None:
    """The n-th market session after d (None past the calendar end): the market-session unit of the embargo."""
    cal = _calendar(calendar)
    i = bisect_right(cal, d) + n - 1
    return cal[i] if 0 <= i < len(cal) else None

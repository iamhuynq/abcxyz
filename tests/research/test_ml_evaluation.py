"""Primary-test statistics of the walk-forward runner (docs/ml-plan.md, PR #13): rank IC with ties, the 30-stock and
60-date thresholds, the training-only baseline sign, identical samples for model and baseline, Newey-West lag."""

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from quant_research import stats
from quant_research.ml import evaluation as ev
from quant_research.ml.contracts import ContractError

D0 = date(2022, 1, 3)


def _days(n):
    return [D0 + timedelta(days=i) for i in range(n)]


def test_rank_ic_is_spearman_with_average_ranks_for_ties():
    x, y = np.array([1.0, 2.0, 2.0, 3.0, 5.0]), np.array([10.0, 30.0, 20.0, 20.0, 50.0])
    rx, ry = np.array([1, 2.5, 2.5, 4, 5]), np.array([1, 4, 2.5, 2.5, 5])
    expected = float(np.corrcoef(rx, ry)[0, 1])
    assert ev.rank_ic(x, y) == pytest.approx(expected, abs=1e-15)
    assert ev.rank_ic(x, y) == pytest.approx(spearmanr(x, y).statistic, abs=1e-15)
    assert ev.rank_ic([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    for a, b in (([1, 1, 1], [1, 2, 3]), ([1, 2, 3], [4, 4, 4]), ([1], [2]), ([1, np.nan], [1, 2])):
        assert math.isnan(ev.rank_ic(a, b))
    with pytest.raises(ContractError):
        ev.rank_ic([1, 2], [1, 2, 3])


def _frame(n_dates, n_stocks, seed=0, signal=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in _days(n_dates):
        b = rng.normal(size=n_stocks)
        y = signal * b + rng.normal(size=n_stocks)
        rows += [(d, b[i], y[i], y[i] + rng.normal()) for i in range(n_stocks)]
    return pd.DataFrame(rows, columns=["date", "baseline", "label", "prediction"])


def test_thirty_stocks_with_a_baseline_make_a_date():
    f = pd.concat([_frame(1, 30), _frame(1, 30).assign(date=D0 + timedelta(days=1))]).reset_index(drop=True)
    f.loc[f.index[-1], "baseline"] = np.nan                                 # second date: 29 with a baseline
    kept = ev.eligible(f)
    assert kept["date"].unique().tolist() == [D0] and len(kept) == 30
    _daily, counts = ev.daily_incremental_ic(f, 1)
    assert counts == {"test_dates": 2, "too_few_stocks_dates": 1, "undefined_ic_dates": 0, "valid_dates": 1}


def test_model_and_baseline_use_identical_rows_and_the_signed_difference():
    f = _frame(3, 35, seed=1)
    f.loc[[0, 1], "baseline"] = np.nan                                       # two rows without a baseline
    for sign in (1, -1):
        daily, _ = ev.daily_incremental_ic(f, sign)
        for _i, r in daily.iterrows():
            g = f[(f["date"] == r["date"]) & np.isfinite(f["baseline"])]
            assert r["n"] == len(g) == (33 if r["date"] == D0 else 35)
            m = ev.rank_ic(g["prediction"], g["label"])
            b = ev.rank_ic(sign * g["baseline"].to_numpy(), g["label"])
            assert (r["model_ic"], r["baseline_ic"], r["incremental_ic"]) == (m, b, m - b)
            assert b == pytest.approx(sign * ev.rank_ic(g["baseline"], g["label"]))
    with pytest.raises(ContractError):
        ev.daily_incremental_ic(f, 0)


def test_a_date_with_an_undefined_ic_is_dropped_for_both_and_counted():
    f = _frame(2, 30, seed=2)
    f.loc[f["date"] == D0, "prediction"] = 0.5                              # constant predictions on one date
    daily, counts = ev.daily_incremental_ic(f, 1)
    assert daily["date"].tolist() == [D0 + timedelta(days=1)] and counts["undefined_ic_dates"] == 1
    f2 = _frame(2, 30, seed=2)
    f2.loc[f2["date"] == D0, "baseline"] = 1.0                               # constant baseline
    assert ev.daily_incremental_ic(f2, 1)[1]["undefined_ic_dates"] == 1


def test_baseline_sign_uses_the_mean_of_daily_ics_and_fails_without_a_clear_sign():
    pos = _frame(60, 30, seed=3, signal=0.5)
    sign, mean, n = ev.baseline_sign(pos)
    assert (sign, n) == (1, 60) and mean == pytest.approx(ev.daily_baseline_ic(pos).mean()) and mean > 0
    assert ev.baseline_sign(pos.assign(label=-pos["label"]))[0] == -1
    with pytest.raises(ContractError, match="59 valid training dates"):
        ev.baseline_sign(pos[pos["date"] != D0])
    # equal weight per date: one large date cannot dominate
    big = pd.concat([_frame(1, 300, seed=4, signal=-3.0), _frame(60, 30, seed=5, signal=0.5).assign(
        date=lambda x: x["date"] + timedelta(days=5))])
    assert ev.baseline_sign(big)[0] == 1
    zero = pos.copy()
    ics = ev.daily_baseline_ic(zero)
    assert len(ics) == 60

    def _flat(train):                                                       # mean exactly 0
        return pd.Series([0.1, -0.1] * 30, dtype=float)
    import quant_research.ml.evaluation as mod
    real = mod.daily_baseline_ic
    mod.daily_baseline_ic = _flat
    try:
        with pytest.raises(ContractError, match="no sign is chosen by default"):
            ev.baseline_sign(zero)
    finally:
        mod.daily_baseline_ic = real


def test_primary_test_is_newey_west_with_lag_horizon_minus_one():
    rng = np.random.default_rng(6)
    x = np.cumsum(rng.normal(size=200)) * 0.001 + rng.normal(size=200) * 0.01
    daily = pd.DataFrame({"date": _days(200), "incremental_ic": x})
    r = ev.primary_test(daily, 10)
    se, t, p, lo, hi = stats.mean_test(x, 9)
    assert (r["se"], r["t"], r["p"], r["ci"], r["lag"], r["n_dates"]) == (se, t, p, [lo, hi], 9, 200)
    assert r["p"] != ev.primary_test(daily, 1)["p"] and r["mean"] == float(x.mean())
    with pytest.raises(ContractError, match="chronological"):
        ev.primary_test(daily.iloc[::-1], 10)
    with pytest.raises(ContractError, match="chronological"):
        ev.primary_test(pd.concat([daily, daily.iloc[-1:]]), 10)
    with pytest.raises(ContractError, match="undefined"):
        ev.primary_test(daily.assign(incremental_ic=0.5), 10)               # zero variance (exact)
    with pytest.raises(ContractError, match="undefined"):
        ev.primary_test(daily.iloc[:2], 10)


def test_primary_label_names_the_target():
    assert ev.primary_label("fwd_ret_exec_5d") == "ml_incremental_rank_ic_fwd_ret_exec_5d"
    assert ev.PRIMARY_TEST in __import__("quant_research.ml.contracts", fromlist=["x"]).PRIMARY_TESTS

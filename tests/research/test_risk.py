"""Risk Engine pure functions: metrics, limits and liquidity on hand-computed values (docs/risk-engine-plan.md)."""

import math

import numpy as np
import pandas as pd
import pytest

from quant_research import exposure
from quant_research.backtest.costs import CostModel
from quant_research.risk import limits, liquidity, metrics
from quant_research.risk.limits import BREACH, INCOMPLETE, PASS, WARNING, Limit, Policy

SQ250 = math.sqrt(250)


def test_drawdown_with_recovery_and_dates():
    dd = metrics.drawdown(np.array([100, 120, 90, 110, 130, 117.0]))
    assert dd.maximum == pytest.approx(-0.25) and (dd.peak_index, dd.trough_index) == (1, 2)
    assert dd.current == pytest.approx(-0.1) and dd.current_peak_index == 4
    assert metrics.drawdown(np.array([100.0])) is None
    assert metrics.drawdown(np.array([100.0, np.nan])) is None and metrics.drawdown(np.array([100.0, 0.0])) is None


def test_volatility_and_downside_volatility_by_hand():
    r = np.array([0.01, -0.02, 0.03, 0.0])
    assert metrics.volatility(r) == pytest.approx(math.sqrt(1.3e-3 / 3) * SQ250)            # mean 0.005
    assert metrics.downside_volatility(r) == pytest.approx(math.sqrt(4e-4 / 3) * SQ250)      # only -0.02 is short
    assert metrics.volatility(np.array([0.01])) is None and metrics.downside_volatility(np.array([])) is None


def test_historical_var_and_es_tail_size():
    r = np.array([-0.05, -0.04, -0.03] + [0.0] * 17)                        # n = 20
    t95 = metrics.historical_var_es(r, 0.95, min_obs=20)                    # k = ceil(20 x 0.05) = 1
    assert (t95.var, t95.es, t95.tail, t95.n) == (pytest.approx(0.05), pytest.approx(0.05), 1, 20)
    t90 = metrics.historical_var_es(r, 0.90, min_obs=20)                    # k = 2
    assert (t90.var, t90.es, t90.tail) == (pytest.approx(0.04), pytest.approx(0.045), 2)
    short = metrics.historical_var_es(r, 0.95)                              # default minimum: 250 returns
    assert short.var is None and short.es is None and short.min_obs == 250
    assert metrics.historical_var_es(r, 0.99).min_obs == 500


def test_hhi_and_effective_positions():
    assert metrics.hhi(np.array([1, 1, 1, 1.0])) == pytest.approx(0.25)
    assert metrics.hhi(np.array([0.7, 0.1, 0.1, 0.1])) == pytest.approx(0.52)
    assert metrics.hhi(np.array([])) is None


def test_exposure_beta_and_risk_shares_reused():
    rng = np.random.default_rng(0)
    m = pd.Series(rng.normal(0, 0.01, 200))
    rets = pd.DataFrame({"A": 1.5 * m + 0.001, "B": rng.normal(0, 0.02, 200)})
    exp = exposure.compute(rets, m, {"A": 1.0, "B": 1.0})
    assert exp.per_stock.set_index("symbol").loc["A", "beta"] == pytest.approx(1.5)
    assert exp.per_stock["risk_share"].sum() == pytest.approx(1.0)


MAX = Limit("pos", "w", "max", 0.10, 0.15)
MIN = Limit("bets", "b", "min", 5.0, 3.0)


@pytest.mark.parametrize("limit, value, status", [
    (MAX, 0.0999, PASS), (MAX, 0.10, WARNING), (MAX, 0.1499, WARNING), (MAX, 0.15, BREACH), (MAX, 0.2, BREACH),
    (MIN, 5.01, PASS), (MIN, 5.0, WARNING), (MIN, 3.01, WARNING), (MIN, 3.0, BREACH), (MIN, 1.0, BREACH)])
def test_limit_thresholds_are_inclusive(limit, value, status):
    assert limits.check(limit, value).status == status


def test_missing_metric_or_data_is_incomplete_never_pass():
    assert limits.check(MAX, None).status == INCOMPLETE
    r = limits.check(MAX, 0.01, missing="no label for XYZ")
    assert r.status == INCOMPLETE and r.reason == "no label for XYZ"
    pol = Policy("p", "v1", (MAX, MIN))
    res = limits.evaluate(pol, {"w": 0.01})
    assert [x.status for x in res] == [PASS, INCOMPLETE]


def test_overall_status_order():
    r = lambda s: limits.LimitResult("x", "m", 0.0, 0, 0, s, "")
    assert limits.overall([r(PASS), r(WARNING)]) == WARNING
    assert limits.overall([r(WARNING), r(BREACH), r(PASS)]) == BREACH
    assert limits.overall([r(BREACH), r(INCOMPLETE)]) == INCOMPLETE
    assert limits.overall([r(PASS)]) == PASS and limits.overall([]) == INCOMPLETE


def test_limit_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        Limit("x", "m", "max", 0.2, 0.1)
    with pytest.raises(ValueError):
        Limit("x", "m", "min", 3.0, 5.0)
    with pytest.raises(ValueError):
        Limit("x", "m", "above", 0.1, 0.2)


@pytest.mark.parametrize("adv", [0, -1e9, float("nan"), float("inf"), None, "x"])
def test_invalid_adv_is_never_liquid(adv):
    assert liquidity.valid_adv(adv) is None
    p = liquidity.position("X", 1e8, adv, 0.002, 0.02, 0, False)
    assert (p.days, p.illiquid, p.cost) == (None, None, None) and "ADV20" in p.reason


def test_liquidation_days_and_cost_by_hand():
    p = liquidity.position("X", 2e8, 1e9, 0.002, 0.02, 0, False)          # 10% of 1 bn per session: 2 sessions
    assert p.days == pytest.approx(2.0) and p.illiquid is True and not p.conditional
    per_side = 0.0015 + 0.001 + 0.002 + 1.0 * 0.02 * math.sqrt(0.1)        # fee + tax + half-spread + impact
    assert p.cost == pytest.approx(2 * 1e8 * per_side) and not p.cost_fallback
    small = liquidity.position("Y", 5e7, 1e9, 0.002, 0.02, 0, False)
    assert small.days == pytest.approx(0.5) and small.illiquid is False
    assert small.cost == pytest.approx(5e7 * (0.0025 + 0.002 + 0.02 * math.sqrt(0.05)))


def test_missing_cost_inputs_use_the_fallback_and_say_so():
    p = liquidity.position("X", 1e8, 1e9, None, None, 0, False)
    m = CostModel()
    assert p.cost_fallback and p.cost == pytest.approx(
        1e8 * (0.0025 + m.fallback_half_spread + m.fallback_sigma * math.sqrt(0.1)))


def test_stale_floor_and_unknown_trade_are_conditional():
    assert liquidity.position("X", 1e8, 1e9, 0.002, 0.02, 5, False).stale is False
    s = liquidity.position("X", 1e8, 1e9, 0.002, 0.02, 6, False)
    assert s.stale is True and s.conditional and "6 sessions" in s.reason
    f = liquidity.position("X", 1e8, 1e9, 0.002, 0.02, 0, True)
    assert f.conditional and "floor" in f.reason
    u = liquidity.position("X", 1e8, 1e9, 0.002, 0.02, None, False)
    assert u.stale is None and u.conditional


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_returns_make_tail_and_volatility_unavailable(bad):
    r = np.array([bad] + [-0.01] * 30)
    t = metrics.historical_var_es(r, 0.95, min_obs=20)
    assert (t.var, t.es) == (None, None) and t.reason == "1 non-finite returns" and t.n == 31
    assert metrics.volatility(r) is None and metrics.downside_volatility(r) is None
    assert metrics.historical_var_es(r[1:], 0.95, min_obs=20).var == pytest.approx(0.01)


def test_threshold_is_inclusive_up_to_float_rounding():
    dd = Limit("dd", "d", "min", -0.20, -0.30)
    assert limits.check(dd, 1.0 / 1.25 - 1).status == WARNING              # -0.19999999999999996
    assert limits.check(dd, -0.1999999).status == PASS
    assert limits.check(MAX, 0.1 - 1e-15).status == WARNING and limits.check(MAX, 0.0999999).status == PASS

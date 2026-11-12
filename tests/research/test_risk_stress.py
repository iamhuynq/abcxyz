"""Stress scenarios on hand-computed positions (docs/risk-engine-pr2-plan.md, D3 to D7)."""

import math

import pandas as pd
import pytest

from quant_research.risk import liquidity, stress
from quant_research.risk.stress import INCOMPLETE, MEASURED

NAV = 1e9


def _pos(**over) -> pd.DataFrame:
    """A 40% beta 1.5 (X, c1), B 30% beta 0.5 (Y, c1), C 20% beta -0.5 (X, c2); 10% cash."""
    df = pd.DataFrame({"symbol": ["A", "B", "C"], "value": [4e8, 3e8, 2e8], "beta": [1.5, 0.5, -0.5],
                       "industry": ["X", "Y", "X"], "cluster": ["c1", "c1", "c2"], "adv": [1e10, 1e10, 1e10],
                       "half_spread": [0.001] * 3, "sigma": [0.01] * 3})
    for k, v in over.items():
        df[k] = v
    df["weight"] = df["value"] / NAV
    base = [liquidity.position(r.symbol, r.value, r.adv, r.half_spread, r.sigma, 0, False) for r in df.itertuples()]
    df["liquidation_cost"] = [x.cost for x in base]
    df["liquidation_days"] = [x.days for x in base]
    df["illiquid"] = [x.illiquid for x in base]
    return df


def _by(results, sid, group=""):
    return next(r for r in results if r.scenario_id == sid and r.group == group)


def test_market_beta_and_uniform_by_hand():
    res = stress.market(_pos(), NAV, (-0.10,))
    beta = _by(res, "market_beta_-10%")
    assert beta.status == MEASURED and beta.valuation_loss == pytest.approx((6e8 + 1.5e8 - 1e8) * 0.10)
    assert beta.valuation_loss_pct == pytest.approx(0.065)
    uni = _by(res, "market_uniform_-10%")
    assert uni.valuation_loss == pytest.approx(9e7) and uni.coverage == pytest.approx(0.9)
    assert "not beta-adjusted" in uni.notes


def test_negative_beta_is_a_gain_and_missing_beta_is_incomplete():
    only_c = _pos().iloc[[2]]
    assert _by(stress.market(only_c, NAV, (-0.10,)), "market_beta_-10%").valuation_loss == pytest.approx(-1e7)
    res = stress.market(_pos(beta=[1.5, None, -0.5]), NAV, (-0.05,))
    b = _by(res, "market_beta_-5%")
    assert b.status == INCOMPLETE and b.valuation_loss is None and b.affected == ("B",) and "no beta for B" in b.reason
    assert _by(res, "market_uniform_-5%").status == MEASURED


def test_industry_shock_hits_one_group_at_a_time():
    res = stress.groups(_pos(), NAV, (-0.20,), "industry", "industry")
    x, y = _by(res, "industry_-20%", "X"), _by(res, "industry_-20%", "Y")
    assert (x.valuation_loss, x.coverage, x.affected) == (pytest.approx(1.2e8), pytest.approx(0.6), ("A", "C"))
    assert (y.valuation_loss, y.affected) == (pytest.approx(6e7), ("B",))
    assert [r.group for r in res] == ["X", "Y"] and all(r.status == MEASURED for r in res)


def test_unclustered_holdings_make_every_cluster_scenario_incomplete():
    res = stress.groups(_pos(cluster=["c1", "c1", None]), NAV, (-0.10,), "cluster", "cluster")
    assert {r.group for r in res} == {"c1"}
    c1 = res[0]
    assert c1.status == INCOMPLETE and c1.valuation_loss == pytest.approx(7e7)
    assert "20.0% of NAV has no cluster: C" in c1.reason
    none = stress.groups(_pos(cluster=[None] * 3), NAV, (-0.10,), "cluster", "cluster")
    assert len(none) == 1 and none[0].status == INCOMPLETE and none[0].valuation_loss is None


def test_liquidity_shock_halves_adv_through_the_cost_model():
    p = _pos()
    r = stress.liquidity_shock(p, NAV, 0.5, liquidity.LiquidityParams())
    assert r.status == MEASURED and r.valuation_loss == 0.0
    a = 4e8 * (0.0015 + 0.001 + 0.001 + 0.01 * math.sqrt(4e8 / 5e9))       # one slice: 10% of 5 bn covers it
    expected = a + sum(3e8 * (0.0025 + 0.001 + 0.01 * math.sqrt(v / 5e9)) for v in (3e8,)) \
        + 2e8 * (0.0025 + 0.001 + 0.01 * math.sqrt(2e8 / 5e9))
    assert r.cost == pytest.approx(expected) and r.cost_before == pytest.approx(p["liquidation_cost"].sum())
    assert r.cost > r.cost_before
    assert "max liquidation days 0.4 -> 0.8" in r.notes


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_adv_makes_the_liquidity_shock_incomplete(bad):
    p = _pos()
    p.loc[1, "adv"] = None if bad != bad else bad
    p["adv"] = p["adv"].map(liquidity.valid_adv)                # as the assessment stores it: valid ADV or None
    r = stress.liquidity_shock(p, NAV, 0.5, liquidity.LiquidityParams())
    assert r.status == INCOMPLETE and r.cost is None and r.affected == ("B",)


def test_suspension_picks_the_largest_with_a_symbol_tie_break_and_writes_down_only_past_60():
    p = _pos(value=[4e8, 2e8, 2e8])                              # B and C tie: B first
    res = stress.suspension(p, NAV, 2, (20, 61))
    short, long_ = _by(res, "suspension_top2_20s"), _by(res, "suspension_top2_61s")
    assert short.affected == long_.affected == ("A", "B")
    assert short.valuation_loss == 0.0 and short.coverage == pytest.approx(0.6) and "exit blocked" in short.notes
    assert long_.valuation_loss == pytest.approx(6e8) and "valued at 0" in long_.notes
    assert stress.suspension(p, NAV, 2, (60,))[0].valuation_loss == 0.0          # the write-down starts after 60


def test_run_marks_everything_incomplete_when_positions_cannot_be_trusted():
    res = stress.run(_pos(), NAV, ["positions without a valid value: Z"])
    assert res and all(r.status == INCOMPLETE and r.reason.startswith("positions without a valid value: Z")
                       for r in res)
    assert stress.run(_pos(), 0.0, []) == [stress.StressResult("all", "all", "", INCOMPLETE, 0.0, None, None, None,
                                                               None, (), "NAV missing or not positive")]


def test_run_covers_every_family_and_never_sums():
    res = stress.run(_pos(), NAV, [])
    fams = {r.family for r in res}
    assert fams == {"market", "industry", "cluster", "liquidity", "suspension"}
    assert len(res) == 6 + 4 + 4 + 1 + 2 and all(r.scenario_id != "total" for r in res)
    assert {r.group for r in res if r.family == "industry"} == {"X", "Y"}
    assert {r.group for r in res if r.family == "cluster"} == {"c1", "c2"}


def test_scenario_set_is_declared_and_serializable():
    s = stress.STRESS_V1
    assert s.key == "stress/v1" and s.market_shocks == (-0.05, -0.10, -0.15) and s.suspension_sessions == (20, 61)
    assert '"adv_factor": 0.5' in s.to_json()

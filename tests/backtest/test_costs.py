"""Cost model v1, benchmarks, matched controls and the economic evaluation."""

import math
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from fireant_crawler.store.warehouse import Warehouse
from quant_research.backtest.benchmarks import NO_KEY, liquidity_weighted_curve, matched_random_selector
from quant_research.backtest.costs import MIN_PAIRS, CostModel
from quant_research.backtest.data import UniverseRule
from quant_research.backtest.engine import run, signal_selector
from quant_research.build import build
from quant_research.daily import FROZEN_STRATEGY
from quant_research.econ import EconRefused, evaluate, strategy_version
from quant_research.econ_report import render
from quant_research.factors import FactorParams
from quant_research.results import ResultsStore
from tests.backtest.test_engine import FEE, TAX, market, random_market, strat
from tests.research.synthetic import build_synthetic_warehouse

NOW = datetime(2026, 10, 9, 20, 0, tzinfo=UTC)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("costs")
    build_synthetic_warehouse(d / "w.duckdb")
    build(d / "w.duckdb", d / "r.duckdb", NOW, FactorParams(min_adv_value=0, min_session_index=0, min_stocks=1))
    return d


def _df(path: Path, sql: str, params: list | None = None) -> pd.DataFrame:
    con = duckdb.connect(str(path), read_only=True)
    try:
        return con.execute(sql, params or []).df()
    finally:
        con.close()


def chl_reference(g: pd.DataFrame) -> pd.Series:
    """Abdi-Ranaldo half-spread per session from the last 21 sessions (pairs k, k+1 <= t), pandas."""
    ok = (g["is_traded"] & ~g["price_jump"] & ~g["bad_source_date"]).fillna(False).astype(bool)
    c, eta = np.log(g["adj_close"]), (np.log(g["adj_high"]) + np.log(g["adj_low"])) / 2
    prod = ((c.shift(1) - eta.shift(1)) * (c.shift(1) - eta)).where(ok & ok.shift(1, fill_value=False))
    mean, n = prod.rolling(20, min_periods=1).mean(), prod.rolling(20, min_periods=1).count()
    return (np.sqrt(np.maximum(4 * mean, 0)) / 2).where(n >= MIN_PAIRS)


def test_tick_floor_and_chl_equal_references(built):
    r = built / "r.duckdb"
    got = _df(r, "SELECT * FROM stock_trading_costs ORDER BY symbol, date")
    panel = _df(r, "SELECT * FROM daily_panel ORDER BY symbol, date")
    tick = np.where(panel["exchange_now"] == "HSX",
                    np.where(panel["close_raw"] < 10, 0.01, np.where(panel["close_raw"] < 50, 0.05, 0.1)), 0.1)
    want_tick = pd.Series(0.5 * tick / panel["close_raw"]).where(panel["close_raw"] > 0)
    np.testing.assert_allclose(got["tick_half_spread"].astype(float), want_tick.astype(float), rtol=1e-12,
                               equal_nan=True)
    want_chl = pd.concat([chl_reference(g) for _, g in panel.groupby("symbol", sort=True)])
    np.testing.assert_allclose(got["chl_half_spread"].astype(float), want_chl.astype(float).to_numpy(),
                               rtol=1e-9, atol=1e-15, equal_nan=True)
    assert got["chl_half_spread"].notna().sum() > 100
    both = got.dropna(subset=["half_spread"])
    assert (both["half_spread"] >= both["tick_half_spread"].fillna(0) - 1e-15).all()


def test_chl_recovers_a_known_spread():
    rng = np.random.default_rng(11)
    spread, days, steps = 0.01, 4000, 200
    mid = np.log(20) + np.cumsum(rng.normal(0, 0.02 / math.sqrt(steps), days * steps)).reshape(days, steps)
    side = rng.choice([-1, 1], days)
    g = pd.DataFrame({"adj_close": np.exp(mid[:, -1] + side * spread / 2),
                      "adj_high": np.exp(mid.max(axis=1) + spread / 2), "adj_low": np.exp(mid.min(axis=1) - spread / 2),
                      "is_traded": True, "price_jump": False, "bad_source_date": False})
    est = chl_reference(g).dropna()
    assert 2 * est.mean() == pytest.approx(spread, rel=0.25)           # full spread recovered on average


def test_costs_are_point_in_time(built, tmp_path):
    cut = date(2023, 9, 1)
    wh = tmp_path / "w.duckdb"
    wh.write_bytes((built / "w.duckdb").read_bytes())
    with Warehouse(wh) as w:
        w.connection.execute("""UPDATE quotes_daily SET price_high = price_high * 1.2, price_close = price_close * 1.1
                                WHERE date > ? AND symbol <> 'VNINDEX'""", [cut])
    build(wh, tmp_path / "r.duckdb", NOW)
    sql = "SELECT * FROM stock_trading_costs WHERE date {} ? ORDER BY symbol, date"
    pd.testing.assert_frame_equal(_df(built / "r.duckdb", sql.format("<="), [cut]),
                                  _df(tmp_path / "r.duckdb", sql.format("<="), [cut]))
    assert not _df(built / "r.duckdb", sql.format(">"), [cut]).equals(_df(tmp_path / "r.duckdb", sql.format(">"), [cut]))


def test_cost_model_formula():
    m = CostModel(k=1.0)
    assert m.side(0.002, 0.02, 1e9, 1e8) == (pytest.approx(0.002 + 0.02 * math.sqrt(0.1)), False)
    assert m.side(0.002, 0.02, 1e9, 0.0) == (0.002, False)
    assert m.side(np.nan, 0.02, 1e9, 1e8)[1] is True                    # missing input: fallback, counted
    assert m.side(0.002, 0.02, np.nan, 1e8) == (pytest.approx(0.002 + 0.02), True)
    assert CostModel(k=2.0, extra_flat=0.001).side(0.0, 0.01, 1e6, 1e6)[0] == pytest.approx(0.021)


def test_engine_without_a_model_is_unchanged_and_zero_inputs_add_nothing():
    m = random_market(5)
    m.half_spread, m.sigma = np.zeros(m.open.shape), np.zeros(m.open.shape)
    m.adv_value[...] = 1e12
    plain, zero = run(m, strat(max_positions=5)), run(m, strat(max_positions=5), costs=CostModel(k=1.0))
    np.testing.assert_array_equal(plain.equity, zero.equity)
    assert plain.costs_paid == zero.costs_paid and zero.cost_fallbacks == 0
    assert strategy_version(FROZEN_STRATEGY, UniverseRule()) == "72c851c7"   # frozen strategy keeps its hash


def test_each_trade_pays_the_v1_formula():
    m = market([10, 10, 11, 11, 12, 12], [10, 10, 11, 11, 12, 12], signal=[1, np.nan, np.nan, np.nan, np.nan, np.nan])
    m.half_spread = np.full((6, 1), 0.003)
    m.sigma = np.full((6, 1), 0.02)
    m.adv_value[...] = 50_000.0
    model = CostModel(k=1.0)
    res = run(m, strat(), costs=model)
    buy_extra = model.side(0.003, 0.02, 50_000.0, 1000.0)[0]                # sized at the budget first
    notional = 1000 / (1 + FEE + buy_extra)
    entry_extra = model.side(0.003, 0.02, 50_000.0, notional)[0]
    shares = notional / 10
    gross = shares * 12
    exit_extra = model.side(0.003, 0.02, 50_000.0, gross)[0]
    assert res.trades[0].costs == pytest.approx(notional * (FEE + entry_extra) + gross * (FEE + TAX + exit_extra))
    leftover = 1000 - notional * (1 + FEE + entry_extra)                   # sized with the budget-level cost
    assert leftover > 0
    assert res.equity[-1] == pytest.approx(leftover + gross * (1 - FEE - TAX - exit_extra))
    assert res.costs_paid == pytest.approx(res.trades[0].costs)


def test_flat_as_model_reproduces_flat_costs():
    from quant_research.econ import FLAT_AS_MODEL
    m = random_market(6)
    m.half_spread = m.sigma = m.tick_half_spread = np.full(m.open.shape, np.nan)        # even with no inputs
    np.testing.assert_array_equal(run(m, strat(max_positions=5)).equity,
                                  run(m, strat(max_positions=5), costs=FLAT_AS_MODEL).equity)


def test_tick_only_spread_uses_the_tick_floor():
    m = market([10, 10, 11, 11, 12, 12], [10, 10, 11, 11, 12, 12], signal=[1, np.nan, np.nan, np.nan, np.nan, np.nan])
    m.half_spread, m.tick_half_spread = np.full((6, 1), 0.01), np.full((6, 1), 0.001)
    m.sigma = np.zeros((6, 1))
    m.adv_value[...] = 1e12
    wide, tick = run(m, strat(), costs=CostModel()), run(m, strat(), costs=CostModel(spread="tick"))
    notional = 1000 / (1 + FEE + 0.001)
    assert tick.trades[0].costs == pytest.approx(notional * (FEE + 0.001) + notional / 10 * 12 * (FEE + TAX + 0.001))
    assert wide.costs_paid > tick.costs_paid


def test_matched_control_keeps_the_key_mix_and_is_deterministic():
    m = random_market(8, n=120, k=60)
    keys = np.array([[f"I{j % 4}" for j in range(60)]] * 120, dtype=object)
    keys[:, 0] = None
    sel = matched_random_selector(3, keys)
    for i in range(0, 120, 7):
        want = sorted(str(keys[i, j]) if keys[i, j] else NO_KEY for j in signal_selector(m, i))
        got = sel(m, i)
        assert sorted(str(keys[i, j]) if keys[i, j] else NO_KEY for j in got) == want
        assert len(set(got)) == len(got) and all(m.universe[i, j] for j in got)
    a = run(m, strat(max_positions=5), matched_random_selector(4, keys))
    b = run(m, strat(max_positions=5), matched_random_selector(4, keys))
    np.testing.assert_array_equal(a.equity, b.equity)


def test_liquidity_weighted_curve_equals_pandas():
    rng = np.random.default_rng(2)
    close = 10 * np.cumprod(1 + rng.normal(0, 0.02, (50, 8)), axis=0)
    close[10, 3] = np.nan
    universe, adv = rng.random((50, 8)) > 0.2, rng.random((50, 8)) * 1e9
    got = liquidity_weighted_curve(close, universe, adv, 100.0)
    want = [100.0]
    for t in range(1, 50):
        r = close[t] / close[t - 1] - 1
        ok = universe[t - 1] & np.isfinite(r) & (adv[t - 1] > 0)
        want.append(want[-1] * (1 + (np.average(r[ok], weights=adv[t - 1][ok]) if ok.any() else 0.0)))
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_evaluate_is_research_only_and_not_logged(built, tmp_path):
    rule = UniverseRule(min_adv_value=0, min_session_index=0, min_stocks_per_date=2, signal_decile=3)
    with ResultsStore(tmp_path / "results.duckdb", built / "r.duckdb") as store:
        for period in ("validation", "holdout", "forward"):
            with pytest.raises(EconRefused):
                evaluate(store, "t", strat(initial_equity=1e9, max_positions=2), period, NOW, rule)
        run_id = evaluate(store, "t", strat(initial_equity=1e9, max_positions=2), "research", NOW, rule,
                          n_control=2, capitals=(1e9,))
        logged = store.con.execute("SELECT count(*) FROM hypothesis_log").fetchone()[0]
        series = {r[0] for r in store.con.execute("SELECT DISTINCT series FROM economic_results").fetchall()}
        metrics = {r[0] for r in store.con.execute("SELECT DISTINCT metric FROM economic_results").fetchall()}
        text = render(store, run_id)
    assert logged == 0
    assert series >= {"strategy", "equal_weight", "liquidity_weighted", "vnindex", "random", "industry_matched",
                      "beta_matched", "writedown_60"}
    assert {"participation_median", "participation_p95"} <= metrics
    assert "## Break-even" in text and "v1_k1" in text
    assert "Valuation policy" in text and "Order size" in text



def test_evaluate_stores_every_control_and_the_exceedance_p(built, tmp_path):
    rule = UniverseRule(min_adv_value=0, min_session_index=0, min_stocks_per_date=2, signal_decile=3)
    with ResultsStore(tmp_path / "results.duckdb", built / "r.duckdb") as store:
        run_id = evaluate(store, "t", strat(initial_equity=1e9, max_positions=2), "research", NOW, rule,
                          n_control=3, capitals=(1e9,))
        outcomes = store.con.execute("""SELECT cost_model, series, count(*), min(seed), max(seed)
                                        FROM control_outcomes WHERE run_id = ? GROUP BY ALL""", [run_id]).fetchall()
        exc = dict(store.con.execute("""SELECT cost_model || '/' || series, value FROM economic_results
                                        WHERE run_id = ? AND metric = 'exceedance_p'""", [run_id]).fetchall())
        share = dict(store.con.execute("""SELECT cost_model || '/' || series, value FROM economic_results
                                          WHERE run_id = ? AND metric = 'share_beaten'""", [run_id]).fetchall())
        median = store.con.execute("""SELECT value FROM economic_results WHERE run_id = ? AND cost_model = 'v1_k1'
                                      AND series = 'random' AND metric = 'median_cagr'""", [run_id]).fetchone()[0]
        cagrs = [r[0] for r in store.con.execute("""SELECT cagr FROM control_outcomes WHERE run_id = ?
                    AND cost_model = 'v1_k1' AND series = 'random' ORDER BY seed""", [run_id]).fetchall()]
    assert sorted(outcomes) == sorted((c, s, 3, 0, 2) for c in ("flat", "v1_k1")
                                      for s in ("random", "industry_matched", "beta_matched"))
    assert set(exc) == {f"{c}/{s}" for c in ("flat", "v1_k1") for s in ("random", "industry_matched", "beta_matched")}
    assert all(v in (0.25, 0.5, 0.75, 1.0) for v in exc.values())                   # (1 + k) / (1 + 3)
    for k, v in exc.items():                                   # no ties here: p = (1 + controls not beaten) / 4
        assert v == pytest.approx((1 + 3 * (1 - share[k])) / 4)
    assert median == pytest.approx(np.median(cagrs))             # the summary comes from the stored outcomes


def test_exceedance_p_counts_ties_against_the_strategy():
    from quant_research.controls import exceedance_p, summary
    assert exceedance_p(10.0, [9.0, 10.0, 11.0]) == pytest.approx(3 / 4)
    assert exceedance_p(10.0, [1.0] * 199) == pytest.approx(1 / 200)                 # never 0
    s = summary(10.0, [9.0, 10.0, 11.0], [0.0, 0.1, 0.2])
    assert s["share_beaten"] == pytest.approx(1 / 3) and s["n_controls"] == 3



def test_control_summary_ignores_invalid_controls_consistently_and_handles_none():
    import warnings

    from quant_research.controls import summary
    nan = float("nan")
    s = summary(2.0, [1.0, nan, 3.0], [0.1, nan, 0.3])
    assert s["n_controls"] == 2 and s["n_invalid"] == 1
    assert s["share_beaten"] == pytest.approx(1 / 2)                   # the NaN control is neither beaten ...
    assert s["exceedance_p"] == pytest.approx((1 + 1) / (1 + 2))       # ... nor beating
    assert s["median_cagr"] == pytest.approx(0.2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                  # no "mean of empty slice"
        empty = summary(2.0, [], [])
        all_bad = summary(2.0, [nan, nan], [nan, nan])
        no_strategy = summary(nan, [1.0, 3.0], [0.1, 0.3])
    for s in (empty, all_bad):
        assert all(s[k] is None for k in ("share_beaten", "exceedance_p", "median_cagr", "p05_cagr", "p95_cagr"))
    assert all_bad["n_invalid"] == 2 and empty["n_controls"] == 0
    assert no_strategy["share_beaten"] is None and no_strategy["exceedance_p"] is None


def test_participation_ignores_orders_without_a_valid_adv():
    from quant_research.backtest.metrics import participation
    out = participation(np.array([10.0, 20.0, 30.0, 40.0]), np.array([100.0, np.nan, 0.0, 400.0]))
    assert out["participation_median"] == pytest.approx(0.1) and out["participation_p95"] == pytest.approx(0.1)
    assert np.isnan(participation(np.array([1.0]), np.array([np.nan]))["participation_median"])

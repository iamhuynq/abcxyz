"""Portfolio construction: ranking buffer, weights and caps, target-weight engine rules, evaluation grid."""

from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

from quant_research.backtest.costs import CostModel
from quant_research.backtest.data import UniverseRule
from quant_research.factors import FactorParams
from quant_research.portfolio.construction import Construction, ranks, select, target_weights
from quant_research.portfolio.engine import FEE, SELL_TAX, rebalance_days, run_targets
from quant_research.portfolio.evaluate import Config, PortfolioRefused, evaluate
from quant_research.portfolio.report import render, verdict
from quant_research.results import ResultsStore
from tests.backtest.test_engine import market as _market
from tests.backtest.test_engine import random_market as _random_market

NOW = datetime(2026, 10, 9, 20, 0, tzinfo=UTC)
BIG_ADV = 1e12            # finite and large: an invalid ADV blocks orders


def market(*args, **kwargs):
    m = _market(*args, **kwargs)
    m.adv_value[...] = BIG_ADV
    return m


def random_market(*args, **kwargs):
    m = _random_market(*args, **kwargs)
    m.adv_value[...] = BIG_ADV
    return m


C2 = Construction(n_names=2, entry_rank=2, exit_rank=2, min_adv_value=0, max_position_weight=1.0,
                  max_industry_weight=1.0)


def flat_scores(values):
    return lambda i: np.asarray(values, float)


def no_industry(m):
    return np.full(m.open.shape, None, dtype=object)


def test_ranking_buffer_keeps_held_names_inside_exit_rank():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, np.nan])
    eligible = np.array([True, True, True, True, False, True])
    r = ranks(scores, eligible)
    assert r.tolist() == [1, 2, 3, 4, 0, 0]
    c = Construction(n_names=2, entry_rank=2, exit_rank=4)
    assert select(r, {3}, c) == [3, 0]                       # held rank 4 kept (buffer), best new name fills
    assert select(r, {3}, replace(c, exit_rank=2)) == [0, 1]  # no buffer: dropped
    assert select(r, {4}, c) == [0, 1]                       # ineligible held name is not kept


def test_weights_slots_and_caps():
    c = Construction(n_names=4, entry_rank=4, exit_rank=4, max_position_weight=0.2, max_industry_weight=0.3)
    ind = np.array(["A", "A", "B", None], dtype=object)
    w = target_weights([0, 1, 2], c, np.ones(4), ind)
    assert w == pytest.approx({0: 0.15, 1: 0.15, 2: 0.2})     # slots 0.25 -> cap 0.2 -> industry A capped at 0.3
    assert sum(w.values()) < 0.75                            # the excess and the empty slot stay cash
    inv = target_weights([0, 2, 3], replace(c, weighting="inverse_vol", max_position_weight=1.0,
                                           max_industry_weight=1.0), np.array([0.01, 1, 0.02, 0.04]), ind)
    assert sum(inv.values()) == pytest.approx(0.75) and inv[0] > inv[2] > inv[3]


def test_rebalance_days():
    dates = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-03-01"))
    dates = dates[np.isin(dates.astype("datetime64[D]").view("int64") % 7, [0, 1, 4, 5, 6])]   # Mon-Fri
    weekly = dates[rebalance_days(dates, "weekly")]
    monthly = dates[rebalance_days(dates, "monthly")]
    assert all((d.astype("datetime64[D]").view("int64") + 3) % 7 == 0 or i == 0 for i, d in enumerate(weekly))
    assert monthly.tolist()[1:] == [np.datetime64("2024-02-01").item()]


def test_trades_at_the_open_after_the_rebalance_close_with_exact_costs():
    # 1970-01-01 is a Thursday: weekly decision closes at 0 and 4 (Monday); trades at the opens of 1 and 5
    m = market(np.full((8, 2), 10.0), np.full((8, 2), 10.0))
    res = run_targets(m, C2, flat_scores([1.0, 0.5]), "weekly", 1000.0, no_industry(m))
    buy_days = sorted({t[0] for t in res.trades})
    assert buy_days == [1, 5]
    first = [t for t in res.trades if t[0] == 1]
    assert [round(t[2] * 10, 6) for t in first] == [500.0, round(min(500.0, (1000 - 500 * (1 + FEE)) / (1 + FEE)), 6)]
    assert res.costs_paid == pytest.approx(sum(t[4] for t in res.trades))


def test_t_plus_2_floor_and_ceiling_rules():
    m = market(np.full((9, 2), 10.0), np.full((9, 2), 10.0), limit_down=np.zeros((9, 2)), open_limit_up=np.zeros((9, 2)))
    scores = {0: [1.0, np.nan], 4: [np.nan, 1.0]}
    run = lambda mk: run_targets(mk, replace(C2, n_names=1, entry_rank=1, exit_rank=1),  # noqa: E731
                                 lambda i: np.array(scores.get(i, [np.nan, np.nan])), "weekly", 1000.0, no_industry(mk))
    res = run(m)
    assert [(t[0], t[1], np.sign(t[2])) for t in res.trades] == [(1, 0, 1), (5, 0, -1), (5, 1, 1)]
    m.limit_down[5, 0] = True                                     # floor at the close of 5: the sale waits
    assert (5, 0) not in [(t[0], t[1]) for t in run(m).trades if t[2] < 0]
    m.limit_down[5, 0], m.open_limit_up[5, 1] = False, True      # ceiling open: the buy is blocked
    assert (5, 1) not in [(t[0], t[1]) for t in run(m).trades]
    # T+2: decision closes on Thu (0), Mon (2) and Mon (5): trades at 1, 3 and 6. The sale wanted at 3 (2 sessions
    # after the buy at 1) is refused; it happens at 6.
    m2 = market(np.full((9, 2), 10.0), np.full((9, 2), 10.0))
    m2.dates = np.array(["1970-01-01", "1970-01-02", "1970-01-05", "1970-01-06", "1970-01-07", "1970-01-12",
                         "1970-01-13", "1970-01-14", "1970-01-15"], dtype="datetime64[D]")
    assert rebalance_days(m2.dates, "weekly").tolist() == [0, 2, 5]
    plan = {0: [1.0, np.nan], 2: [np.nan, 1.0], 5: [np.nan, 1.0]}
    r2 = run_targets(m2, replace(C2, n_names=1, entry_rank=1, exit_rank=1), lambda i: np.array(plan[i]),
                     "weekly", 1000.0, no_industry(m2))
    sells = [(t[0], t[1]) for t in r2.trades if t[2] < 0]
    assert sells == [(6, 0)] and r2.orders_blocked_t2 == 1
    r3 = run_targets(m2, replace(C2, n_names=1, entry_rank=1, exit_rank=1), lambda i: np.array(plan[i]),
                     "weekly", 1000.0, no_industry(m2), sell_lag=2)              # optimistic: sale at 3 allowed
    assert (3, 0) in [(t[0], t[1]) for t in r3.trades if t[2] < 0] and r3.orders_blocked_t2 == 0


def test_participation_cap_and_v1_costs():
    m = market(np.full((6, 1), 10.0), np.full((6, 1), 10.0))
    m.adv_value[...] = 2000.0
    m.half_spread, m.sigma = np.full((6, 1), 0.002), np.full((6, 1), 0.02)
    c = replace(C2, n_names=1, entry_rank=1, exit_rank=1, max_participation=0.1)
    res = run_targets(m, c, flat_scores([1.0]), "weekly", 1000.0, no_industry(m), costs=CostModel(k=1.0))
    first = res.trades[0]
    assert first[2] * first[3] == pytest.approx(200.0) and res.orders_capped >= 1     # 10% of ADV 2000
    assert first[4] == pytest.approx(200.0 * (FEE + 0.002 + 0.02 * np.sqrt(0.1)))


def test_accounting_reconciles_and_is_point_in_time():
    m = random_market(4, n=200, k=30)
    ind = np.array([[f"I{j % 3}" for j in range(30)]] * 200, dtype=object)
    sc = np.where(np.isnan(m.signal), np.random.default_rng(1).random(m.signal.shape), m.signal)
    c = Construction(n_names=5, entry_rank=5, exit_rank=10, min_adv_value=0, max_industry_weight=0.5)
    res = run_targets(m, c, lambda i: sc[i], "weekly", 1e6, ind)
    flows = sum(-t[2] * t[3] - t[4] for t in res.trades)
    held = {}
    for t in res.trades:
        held[t[1]] = held.get(t[1], 0.0) + t[2]
    last = {j: m.close[np.flatnonzero(np.isfinite(m.close[:, j]))[-1], j] for j in held}
    value = sum(s * last[j] for j, s in held.items() if s > 1e-9)
    assert res.equity[-1] == pytest.approx(1e6 + flows + value, rel=1e-9)
    assert res.max_target_industry_weight <= 0.5 + 1e-12
    cut = 100
    m2 = random_market(4, n=200, k=30)
    m2.close[cut + 1:] *= 1.5
    m2.open[cut + 1:] *= 0.7
    sc2 = sc.copy()
    sc2[cut + 1:] = np.random.default_rng(9).random(sc2[cut + 1:].shape)
    res2 = run_targets(m2, c, lambda i: sc2[i], "weekly", 1e6, ind)
    np.testing.assert_array_equal(res.equity[:cut + 1], res2.equity[:cut + 1])
    assert not np.array_equal(res.equity, res2.equity)


def test_evaluate_grid_rules(tmp_path_factory):
    from quant_research.build import build
    from tests.research.synthetic import build_synthetic_warehouse
    d = tmp_path_factory.mktemp("pf")
    build_synthetic_warehouse(d / "w.duckdb")
    build(d / "w.duckdb", d / "r.duckdb", NOW, FactorParams(min_adv_value=0, min_session_index=0, min_stocks=1))
    configs = {"T1": Config("momentum_1m", "weekly", replace(C2, n_names=2)),
               "T2": Config("momentum_1m", "weekly", replace(C2, n_names=2), ("direction", "Bear"))}
    rule = UniverseRule(min_adv_value=0, min_session_index=0, min_stocks_per_date=1, signal_decile=3)
    with ResultsStore(d / "results.duckdb", d / "r.duckdb") as store:
        for period in ("validation", "holdout", "forward"):
            with pytest.raises(PortfolioRefused):
                evaluate(store, "T1", NOW, period, configs, rule)
        runs = {n: evaluate(store, n, NOW.replace(minute=i), "research", configs, rule, capitals=(1e9,), n_control=2)
                for i, n in enumerate(configs)}
        with pytest.raises(PortfolioRefused, match="invalidate"):
            evaluate(store, "T1", NOW.replace(hour=21), "research", configs, rule, capitals=(1e9,), n_control=2)
        logged = store.con.execute("SELECT run_id, label FROM hypothesis_log").fetchall()
        extra = {r[0] for r in store.con.execute("""SELECT DISTINCT series || '/' || metric FROM portfolio_results
                                                   WHERE run_id = ?""", [runs["T1"]]).fetchall()}
        outcomes = store.con.execute("""SELECT count(*), count(DISTINCT seed), any_value(series) FROM control_outcomes
                                        WHERE run_id = ?""", [runs["T1"]]).fetchone()
        text = render(store, runs)
        cash = store.con.execute("""SELECT value FROM portfolio_results WHERE run_id = ? AND metric = 'avg_cash_share'
                                    AND cost_model = 'flat'""", [runs["T2"]]).fetchone()[0]
    assert sorted(logged) == sorted((r, "portfolio_excess_vs_equal_weight") for r in runs.values())
    assert "| T1 |" in text and "Worth a pre-registration" in text
    assert {"writedown_60/cagr", "writedown_60/break_even_extra_cost_per_side", "strategy/participation_median",
            "sell_lag_2/cagr", "random_construction/exceedance_p"} <= extra
    assert outcomes == (2, 2, "random_construction")                  # every control stored (n_control=2)
    assert cash > 0                                               # Bear filter holds cash part of the time


def test_reading_rule():
    wd = "writedown_60"
    base = {(1e9, "v1_k1", "strategy", "cagr"): 0.05, (1e9, "v1_k1", wd, "cagr"): 0.04,
            (1e9, "gross", "equal_weight", "cagr"): 0.01,
            (1e9, "v1_k1", "random_construction", "share_beaten"): 0.95,
            (1e9, "v1_k1", "strategy", "break_even_extra_cost_per_side"): 0.003,
            (1e9, "v1_k1", wd, "break_even_extra_cost_per_side"): 0.002}
    assert verdict(base.get)[0] is True
    for key, bad in (((1e9, "v1_k1", "strategy", "cagr"), 0.0), ((1e9, "v1_k1", "random_construction", "share_beaten"), 0.9),
                     ((1e9, "v1_k1", "strategy", "break_even_extra_cost_per_side"), 0.0019)):
        assert verdict({**base, key: bad}.get)[0] is False


def test_reading_rule_uses_the_worse_valuation_and_fails_without_it():
    wd = "writedown_60"
    base = {(1e9, "v1_k1", "strategy", "cagr"): 0.05, (1e9, "v1_k1", wd, "cagr"): 0.04,
            (1e9, "gross", "equal_weight", "cagr"): 0.01,
            (1e9, "v1_k1", "random_construction", "share_beaten"): 1.0,
            (1e9, "v1_k1", "strategy", "break_even_extra_cost_per_side"): 0.003,
            (1e9, "v1_k1", wd, "break_even_extra_cost_per_side"): 0.002}
    assert verdict(base.get)[0] is True
    assert verdict({**base, (1e9, "v1_k1", wd, "cagr"): 0.005}.get)[0] is False                # passes only by default
    assert verdict({**base, (1e9, "v1_k1", wd, "break_even_extra_cost_per_side"): 0.001}.get)[0] is False
    missing = {k: v for k, v in base.items() if k[2] != wd}
    ok, notes = verdict(missing.get)
    assert ok is False and any("write-down valuation missing" in n for n in notes)


@pytest.mark.parametrize("target", [0.9, 0.4, 0.0])
def test_persistent_random_scores_match_the_target_rank_persistence(target):
    from quant_research.portfolio.evaluate import persistent_random_scores, rank_persistence
    z = persistent_random_scores(3, 400, 150, target)
    assert rank_persistence(z, np.arange(150)) == pytest.approx(target, abs=0.03)
    np.testing.assert_array_equal(z, persistent_random_scores(3, 400, 150, target))      # deterministic per seed


def test_matched_control_trades_about_as_much_as_a_persistent_signal():
    from quant_research.portfolio.evaluate import _metrics, persistent_random_scores, rank_persistence
    m = random_market(12, n=260, k=120)
    m.universe[...] = True
    days = rebalance_days(m.dates, "weekly")
    pos = {int(d): r for r, d in enumerate(days)}
    signal = persistent_random_scores(100, 120, len(days), 0.9)                         # a persistent "signal"
    rho = rank_persistence(signal, np.arange(len(days)))
    c = Construction(n_names=10, entry_rank=10, exit_rank=20, min_adv_value=-1)
    run = lambda fn: _metrics(run_targets(m, c, fn, "weekly", 1e6, no_industry(m)), 1e6)["turnover_per_year"]  # noqa: E731
    strategy = run(lambda i: signal[pos[i]])
    matched = np.median([run(lambda i, z=persistent_random_scores(s, 120, len(days), rho): z[pos[i]])
                         for s in range(5)])
    fresh = np.median([run(lambda i, s=s: np.random.default_rng([s, i]).random(120)) for s in range(5)])
    assert 0.7 < matched / strategy < 1.4 and fresh > 2 * strategy


def test_data_error_exit_pays_the_same_costs_as_any_sale():
    m = market(np.full((8, 1), 10.0), np.full((8, 1), 10.0))
    m.half_spread, m.sigma = np.full((8, 1), 0.003), np.full((8, 1), 0.02)
    m.adv_value[...] = 1e5
    m.price_jump[3, 0] = True                                       # data error at the close of 3
    model = CostModel(k=1.0)
    res = run_targets(m, replace(C2, n_names=1, entry_rank=1, exit_rank=1), flat_scores([1.0]), "monthly", 1000.0,
                      no_industry(m), costs=model)
    exit_trade = [t for t in res.trades if t[2] < 0][0]
    value = -exit_trade[2] * exit_trade[3]
    assert exit_trade[0] == 3 and res.data_error_exits == 1
    assert exit_trade[4] == pytest.approx(value * (FEE + SELL_TAX + model.side(0.003, 0.02, 1e5, value)[0]))
    buys = sum(t[2] * t[3] + t[4] for t in res.trades if t[2] > 0)
    assert res.equity[-1] == pytest.approx(1000.0 - buys + value - exit_trade[4])        # cash reconciles


def test_actual_industry_weight_is_measured_after_trades_and_drift():
    n = 12
    close = np.full((n, 2), 10.0)
    close[3:, 0] = 30.0                                             # industry A triples after the buy
    m = market(close.copy(), close)
    m.limit_down[5, 0] = True                                       # the trim at the next rebalance is blocked
    ind = np.array([["A", "B"]] * n, dtype=object)
    c = replace(C2, max_industry_weight=0.5)
    res = run_targets(m, c, flat_scores([1.0, 0.5]), "weekly", 1000.0, ind)
    assert res.max_target_industry_weight <= 0.5 + 1e-12
    assert res.max_actual_industry_weight > 0.7                     # about 0.75 after the rise
    assert res.industry_cap_breach_sessions >= 1 and res.industry_cap_breach_after_trades >= 1


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0.0])
def test_invalid_adv_blocks_orders_instead_of_removing_the_cap(bad):
    m = market(np.full((6, 1), 10.0), np.full((6, 1), 10.0))
    m.adv_value[...] = bad
    res = run_targets(m, replace(C2, n_names=1, entry_rank=1, exit_rank=1, min_adv_value=-1), flat_scores([1.0]),
                      "weekly", 1000.0, no_industry(m))
    assert res.trades == [] and res.orders_blocked_no_adv >= 1 and res.equity[-1] == 1000.0

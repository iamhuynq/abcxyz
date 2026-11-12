"""Risk evaluation end to end on a synthetic store: paper snapshot, backtest snapshot, persistence, report, CLI,
period gate, frozen policies, and no change to anything that existed before."""

from datetime import UTC, date, datetime, timedelta

import duckdb
import numpy as np
import pytest

from quant_research import cli, daily
from quant_research.backtest import runner
from quant_research.backtest.metrics import drawdown as stored_drawdown
from quant_research.build import build
from quant_research.factors import FactorParams
from quant_research.regimes import RegimeParams
from quant_research.results import ResultsStore
from quant_research.risk import data, evaluate, limits, report
from tests.research.synthetic import DELISTED_LAST_TRADE, build_synthetic_warehouse, calendar

NOW = datetime(2026, 11, 11, 9, 0, tzinfo=UTC)
START = date(2022, 1, 3)                   # 300 sessions inside the research period
DAYS = calendar(START)
VAL = 279                                  # valuation session index; DEL stopped trading at 200 (79 sessions ago)


@pytest.fixture
def env(tmp_path, monkeypatch):
    d = tmp_path / "data"
    d.mkdir()
    build_synthetic_warehouse(d / "warehouse.duckdb", start=START)
    build(d / "warehouse.duckdb", d / "research.duckdb", NOW,
          FactorParams(min_adv_value=0, min_session_index=0, min_stocks=1, min_industry=2),
          RegimeParams(window=100, min_obs=50, breadth_min_adv=0))
    env_file = tmp_path / ".env"
    env_file.write_text("FIREANT_TOKEN=\n")
    monkeypatch.delenv("FIREANT_TOKEN", raising=False)
    monkeypatch.setenv("DATA_DIR", str(d))
    monkeypatch.setenv("ENV_FILE", str(env_file))
    with ResultsStore(d / "results.duckdb", d / "research.duckdb") as store:
        store.con.execute(daily.SCHEMA)
        store.con.execute(runner.SCHEMA)
        _paper(store)
    return d


def _paper(store: ResultsStore) -> None:
    """NAV 1 bn: STK 40%, UPC 30%, DEL 20% (stopped trading), cash 10%. The curve peaks at 1.25 bn."""
    run_date = DAYS[VAL]
    curve = np.concatenate([np.linspace(1.0e9, 1.25e9, 11), np.linspace(1.25e9, 1.0e9, 20)[1:]])
    store.con.executemany("INSERT INTO paper_daily (run_date, date, equity, n_positions, equity_traded_mark) "
                          "VALUES (?, ?, ?, 3, ?)",
                          [(run_date, DAYS[VAL - 29 + i], float(e), float(e)) for i, e in enumerate(curve)])
    store.con.executemany("INSERT INTO paper_positions VALUES (?, ?, ?, ?, ?, ?)",
                          [(run_date, "STK", DAYS[150], 2e7, 19.0, 20.0), (run_date, "UPC", DAYS[150], 3e7, 9.0, 10.0),
                           (run_date, "DEL", DAYS[150], 4e7, 5.0, 5.0)])
    store.con.execute("INSERT INTO paper_state VALUES (1, ?, ?)", [run_date, NOW])


def _snapshot_tables(path) -> dict:
    with duckdb.connect(str(path), read_only=True) as con:
        return {t: con.execute(f"SELECT * FROM {t} ORDER BY ALL").fetchall()
                for t in ("paper_daily", "paper_positions", "paper_state", "hypothesis_log", "hypotheses",
                          "hypothesis_transitions", "backtest_daily", "backtest_trades")}


def _status(a) -> dict:
    return {r.metric: r.status for r in a.limit_results}


def test_paper_snapshot_evaluation(env):
    before = _snapshot_tables(env / "results.duckdb")
    with ResultsStore(env / "results.duckdb", env / "research.duckdb") as store:
        snap = data.paper_snapshot(store.con)
        assert snap.valuation_date == DAYS[VAL] and snap.nav == pytest.approx(1e9) and snap.cash == pytest.approx(1e8)
        run_id, a = evaluate.evaluate(store, snap, limits.DEFAULT_V1, NOW)
        text = report.render(store, run_id)
        kinds = store.con.execute("SELECT kind, status FROM research_runs").fetchall()
    st = _status(a)
    assert a.status == limits.INCOMPLETE
    assert st["max_position_weight"] == st["max_industry_weight"] == limits.BREACH
    assert st["max_cluster_weight"] == limits.INCOMPLETE              # no cross database: no cluster label
    assert st["abs_beta"] == st["effective_bets"] == limits.INCOMPLETE  # DEL has no returns in the window
    assert st["illiquid_weight"] == limits.INCOMPLETE                   # DEL has no valid ADV on the date
    assert st["stale_weight"] == limits.BREACH and st["max_drawdown"] == limits.BREACH
    by = {r.metric: r for r in a.limit_results}
    assert by["max_position_weight"].affected == ("DEL", "STK", "UPC") and by["max_position_weight"].value == 0.4
    assert "DEL" in by["abs_beta"].reason and "DEL" in by["illiquid_weight"].reason
    # curve drawdown -20%; the write-down of DEL (79 > 60 sessions without a trade) gives 0.8 / 1.25 - 1 = -36%
    assert by["max_drawdown"].value == pytest.approx(0.8 / 1.25 - 1)
    m = {(s, k): v for s, k, v, _n in a.metrics}
    assert m[("curve:pre-registered marks", "max_drawdown")] == pytest.approx(-0.2)
    assert m[("curve:pre-registered marks", "var_95_1d")] is None          # 30 returns < 250
    stale = a.positions.set_index("symbol")
    assert stale.loc["DEL", "stale_sessions"] == VAL - DELISTED_LAST_TRADE and bool(stale.loc["DEL", "stale"])
    assert "## Status: INCOMPLETE" in text and "policy `default/v1` (provisional)" in text and "Provenance:" in text
    assert kinds == [("risk", "ok")]
    assert _snapshot_tables(env / "results.duckdb") == before         # nothing that existed was changed


def test_stored_evaluations_survive_new_runs_and_policies_are_frozen(env):
    with ResultsStore(env / "results.duckdb", env / "research.duckdb") as store:
        snap = data.paper_snapshot(store.con)
        first, _ = evaluate.evaluate(store, snap, limits.DEFAULT_V1, NOW)
        rows = store.con.execute("SELECT * FROM risk_limit_results WHERE run_id = ? ORDER BY ALL", [first]).fetchall()
        changed = limits.Policy("default", "v1", limits.DEFAULT_V1.limits[:1])
        with pytest.raises(ValueError, match="new version"):
            evaluate.evaluate(store, snap, changed, NOW + timedelta(minutes=1))
        v2 = limits.Policy("default", "v2", limits.DEFAULT_V1.limits[:1])
        second, a2 = evaluate.evaluate(store, snap, v2, NOW + timedelta(minutes=2))
        assert len(a2.limit_results) == 1 and second != first
        assert store.con.execute("SELECT * FROM risk_limit_results WHERE run_id = ? ORDER BY ALL",
                                 [first]).fetchall() == rows
        assert store.con.execute("SELECT count(*) FROM risk_policies").fetchone()[0] == 2


def test_backtest_snapshot_rebuilds_positions_and_the_stored_drawdown(env):
    with ResultsStore(env / "results.duckdb", env / "research.duckdb") as store:
        stk = dict(store.con.execute("SELECT date, adj_close FROM rs.daily_panel WHERE symbol = 'STK'").fetchall())
        days = DAYS[100:151]
        equity = np.array([1e9 * (1 + 0.1 * np.sin(i / 5)) for i in range(len(days))])
        shares = 1e7
        cash = equity - shares * np.array([stk[d] for d in days])
        store.con.executemany("INSERT INTO backtest_daily VALUES ('BT', ?, ?, ?, 0, 1, NULL, NULL, NULL)",
                              [(d, float(e), float(c)) for d, e, c in zip(days, equity, cash)])
        store.con.executemany("INSERT INTO backtest_trades VALUES ('BT', ?, ?, ?, 1, 1, ?, 0, 0, 0, ?)",
                              [("STK", DAYS[100], DAYS[150], shares, "open_at_end"),
                               ("UPC", DAYS[110], DAYS[140], 5e6, "planned")])
        last = data.backtest_snapshot(store.con, "BT")
        mid = data.backtest_snapshot(store.con, "BT", DAYS[130])
        assert list(last.positions["symbol"]) == ["STK"] and sorted(mid.positions["symbol"]) == ["STK", "UPC"]
        assert list(data.backtest_snapshot(store.con, "BT", DAYS[140]).positions["symbol"]) == ["STK"]   # sold that close
        assert sorted(data.backtest_snapshot(store.con, "BT", DAYS[139]).positions["symbol"]) == ["STK", "UPC"]
        assert "differ from the stored equity by 0 VND" in last.notes[0]
        run_id, a = evaluate.evaluate(store, last, limits.DEFAULT_V1, NOW)
        m = {(s, k): v for s, k, v, _n in a.metrics}
        assert m[("curve:stored equity", "max_drawdown")] == pytest.approx(stored_drawdown(equity)[0])
        assert data.backtest_snapshot(store.con, "missing") is None


def test_frozen_periods_are_refused(env):
    with ResultsStore(env / "results.duckdb", env / "research.duckdb") as store:
        snap = data.paper_snapshot(store.con)
        for day in (date(2024, 6, 3), date(2026, 5, 4)):              # validation, holdout
            snap.valuation_date = day
            with pytest.raises(ValueError, match="frozen"):
                evaluate.evaluate(store, snap, limits.DEFAULT_V1, NOW)
        assert store.con.execute("SELECT count(*) FROM research_runs").fetchone()[0] == 0


def test_cli_evaluates_and_rerenders(env, capsys):
    assert cli.main(["risk", "evaluate", "--paper"]) == 0
    out = capsys.readouterr().out
    assert f"paper:{DAYS[VAL]} at {DAYS[VAL]}: INCOMPLETE" in out and "BREACH" in out
    report_file = next((env / "reports" / "research").glob("risk-*.md"))
    run_id = report_file.stem.removeprefix("risk-")
    first = report_file.read_text()
    assert cli.main(["risk", "report", run_id]) == 0
    assert report_file.read_text() == first                            # re-rendered from the stored rows
    assert cli.main(["risk", "evaluate", "--backtest", "nothing"]) == 1
    assert cli.main(["risk", "policies"]) == 0
    capsys.readouterr()


def test_clusters_are_point_in_time(env, tmp_path):
    cx = tmp_path / "cross.duckdb"
    with duckdb.connect(str(cx)) as con:
        con.execute("CREATE TABLE symbols (id INTEGER, symbol VARCHAR)")
        con.execute("INSERT INTO symbols VALUES (1, 'STK'), (2, 'UPC')")
        con.execute("CREATE TABLE clusters_monthly (month_end TIMESTAMP, symbol_id INTEGER, cluster INTEGER, "
                    "industry_l2_code VARCHAR)")
        con.execute("""INSERT INTO clusters_monthly VALUES ('2023-01-31', 1, 4, NULL), ('2023-01-31', 2, 4, NULL),
                       ('2023-03-31', 1, 7, NULL), ('2023-03-31', 2, 8, NULL)""")
    with ResultsStore(env / "results.duckdb", env / "research.duckdb") as store:
        assert data.clusters(store.con, ("STK",), date(2023, 2, 15)) == {}         # no cx attached: no labels
        store.con.execute(f"ATTACH '{cx.as_posix()}' AS cx (READ_ONLY)")
        assert data.clusters(store.con, ("STK", "UPC"), date(2023, 2, 15)) == {"STK": "cluster 4", "UPC": "cluster 4"}
        assert data.clusters(store.con, ("STK", "UPC"), date(2023, 3, 31)) == {"STK": "cluster 7", "UPC": "cluster 8"}
        assert data.clusters(store.con, ("STK",), date(2022, 12, 30)) == {}


def _inputs(symbols):
    import pandas as pd
    return pd.DataFrame({"symbol": symbols, "adv": 1e12, "half_spread": 0.001, "sigma": 0.01, "industry": "X",
                         "last_trade": date(2023, 5, 2), "stale_sessions": 0, "at_floor": False})


def _returns(betas: dict[str, float], n: int = 120):
    import pandas as pd
    rng = np.random.default_rng(1)
    m = pd.Series(rng.normal(0, 0.01, n))
    return {"returns": pd.DataFrame({s: b * m + rng.normal(0, 0.001, n) for s, b in betas.items()}), "market": m}


@pytest.mark.parametrize("nav", [0.0, -1e9, float("nan")])
def test_non_positive_nav_is_incomplete_everywhere(nav):
    import pandas as pd
    snap = data.Snapshot("x", date(2023, 5, 2), nav, 0.0, pd.DataFrame({"symbol": ["A"], "value": [1.0]}),
                         {"c": (np.arange(3), np.array([1.0, 1.1, 1.0]))})
    a = evaluate.assess(snap, _inputs(["A"]), _returns({"A": 1.0}), {"A": "c1"}, None, limits.DEFAULT_V1)
    assert a.status == limits.INCOMPLETE and {r.status for r in a.limit_results} == {limits.INCOMPLETE}


def test_cash_dilutes_beta_and_volatility():
    """Positions with beta 1.6 at half the NAV: the portfolio beta is 0.8 (PASS), not 1.6 (BREACH)."""
    import pandas as pd
    syms = [f"S{i}" for i in range(10)]
    snap = data.Snapshot("x", date(2023, 5, 2), 2e9, 1e9, pd.DataFrame({"symbol": syms, "value": [1e8] * 10}),
                         {"c": (np.arange(3), np.array([2e9, 2.1e9, 2e9]))})
    a = evaluate.assess(snap, _inputs(syms), _returns({s: 1.6 for s in syms}), {s: f"c{i}" for i, s in enumerate(syms)},
                        None, limits.DEFAULT_V1)
    beta = next(r for r in a.limit_results if r.metric == "abs_beta")
    assert beta.status == limits.PASS and beta.value == pytest.approx(0.8, rel=0.02)


def _snap(values: dict[str, float], nav: float, curve=(1.0, 1.1, 1.0), gap=None, prices=None):
    import pandas as pd
    p = pd.DataFrame({"symbol": list(values), "value": list(values.values())})
    p["price"] = [(prices or {}).get(s, 10.0) for s in values]
    p["shares"] = p["value"] / p["price"]
    return data.Snapshot("x", date(2023, 5, 2), nav, nav - p["value"].sum(), p,
                         {"c": (np.arange(len(curve)), np.array(curve) * nav)}, [], gap)


def _limits(a) -> dict:
    return {r.metric: r for r in a.limit_results}


def test_unclustered_holdings_are_reported_and_keep_the_limit_incomplete():
    snap = _snap({"A": 3e8, "B": 2e8, "C": 4e8}, 1e9)
    a = evaluate.assess(snap, _inputs(["A", "B", "C"]), _returns({"A": 1, "B": 1, "C": 1}),
                        {"A": "c1", "B": "c2"}, None, limits.DEFAULT_V2)
    cl = _limits(a)["max_cluster_weight"]
    assert cl.status == limits.INCOMPLETE and cl.value == pytest.approx(0.3)
    assert "40.0% of NAV has no cluster: C" in cl.reason
    m = {k: v for s, k, v, _n in a.metrics if s == "concentration"}
    assert m["cluster_unlabelled_weight"] == pytest.approx(0.4)
    assert m["max_cluster_weight_unlabelled_as_one"] == pytest.approx(0.4)
    assert m["max_cluster_weight_labelled"] == pytest.approx(0.3)


def test_v2_checks_historical_and_write_down_drawdowns_separately(env):
    with ResultsStore(env / "results.duckdb", env / "research.duckdb") as store:
        _run_id, a = evaluate.evaluate(store, data.paper_snapshot(store.con), limits.DEFAULT_V2, NOW)
    by = _limits(a)
    assert by["max_drawdown_historical"].value == pytest.approx(-0.2)
    assert by["max_drawdown_historical"].status == limits.WARNING                    # -20% is the warning level
    assert by["current_drawdown_write_down"].value == pytest.approx(0.8 / 1.25 - 1)
    assert by["current_drawdown_write_down"].status == limits.BREACH
    assert "max_drawdown" not in by


def test_unvalued_positions_make_position_limits_incomplete_not_history():
    snap = _snap({"A": 3e8, "B": 2e8}, 1e9, prices={"B": float("nan")})
    a = evaluate.assess(snap, _inputs(["A", "B"]), _returns({"A": 1, "B": 1}), {"A": "c1", "B": "c2"}, None,
                        limits.DEFAULT_V2)
    by = _limits(a)
    assert by["max_drawdown_historical"].status != limits.INCOMPLETE
    for metric in ("max_position_weight", "max_industry_weight", "abs_beta", "illiquid_weight",
                   "current_drawdown_write_down"):
        assert by[metric].status == limits.INCOMPLETE and "without a valid value: B" in by[metric].reason, metric


@pytest.mark.parametrize("gap, status", [(0.009e9, limits.PASS), (0.011e9, limits.INCOMPLETE),
                                         (float("nan"), limits.INCOMPLETE)])
def test_reconciliation_gap_above_one_percent_is_incomplete(gap, status):
    snap = _snap({f"S{i}": 0.09e9 for i in range(10)}, 1e9, gap=gap)
    a = evaluate.assess(snap, _inputs(list(snap.positions["symbol"])), _returns({s: 1 for s in snap.positions["symbol"]}),
                        {s: f"c{i}" for i, s in enumerate(snap.positions["symbol"])}, None, limits.DEFAULT_V2)
    assert _limits(a)["max_position_weight"].status == status
    assert _limits(a)["max_drawdown_historical"].status != limits.INCOMPLETE


def test_cost_fallback_weight_is_reported():
    snap = _snap({"A": 3e8, "B": 2e8}, 1e9)
    inputs = _inputs(["A", "B"])
    inputs.loc[inputs["symbol"] == "B", ["half_spread", "sigma"]] = None
    a = evaluate.assess(snap, inputs, _returns({"A": 1, "B": 1}), {"A": "c1", "B": "c2"}, None, limits.DEFAULT_V2)
    m = {k: v for s, k, v, _n in a.metrics if s == "liquidity"}
    assert m["cost_fallback_weight"] == pytest.approx(0.2) and m["cost_fallbacks"] == 1


@pytest.mark.parametrize("nulls, expected", [(0, None), (1, "1 of 30 values missing or not finite"),
                                             (30, "30 of 30 values missing or not finite")])
def test_a_missing_traded_mark_makes_the_historical_drawdown_incomplete(env, nulls, expected):
    """The traded-mark curve is required for the paper portfolio: a gap is reported, never dropped."""
    with ResultsStore(env / "results.duckdb", env / "research.duckdb") as store:
        dates = [d for (d,) in store.con.execute("SELECT date FROM paper_daily ORDER BY date").fetchall()]
        hole = dates[10:10 + nulls] if nulls == 1 else dates[:nulls]
        for d in hole:
            store.con.execute("UPDATE paper_daily SET equity_traded_mark = NULL WHERE date = ?", [d])
        snap = data.paper_snapshot(store.con)
        assert set(snap.curves) == {"pre-registered marks", "traded marks"}
        _run_id, a = evaluate.evaluate(store, snap, limits.DEFAULT_V2, NOW)
    hist = _limits(a)["max_drawdown_historical"]
    m = {(s, k): (v, n) for s, k, v, n in a.metrics}
    if expected is None:
        assert hist.status == limits.WARNING and hist.value == pytest.approx(-0.2)
        assert m[("curve:traded marks", "max_drawdown")][0] == pytest.approx(-0.2)
    else:
        assert hist.status == limits.INCOMPLETE
        assert hist.reason == f"valuation curve unavailable: traded marks ({expected})"
        assert m[("curve:traded marks", "max_drawdown")][0] is None
        assert m[("curve:traded marks", "current_drawdown")][1] == expected
        assert m[("curve:pre-registered marks", "max_drawdown")][0] == pytest.approx(-0.2)   # still shown
    assert _limits(a)["current_drawdown_write_down"].status == limits.BREACH                  # other limits unaffected

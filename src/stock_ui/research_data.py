"""Queries over results.duckdb for the Research, Backtest and Daily pages (read-only, bound parameters)."""

import duckdb
import pandas as pd

from quant_research import paper_state
from stock_ui.db import Reader, ReadResult

LIFT_HORIZON = "exec_excess_10d"


def _has(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = ?", [table]).fetchone()[0])


def _df_or_none(con: duckdb.DuckDBPyConnection, table: str, sql: str, params: list | None = None) -> pd.DataFrame | None:
    return con.execute(sql, params or []).df() if _has(con, table) else None


# Research -------------------------------------------------------------------------------------

def runs(reader: Reader) -> ReadResult:
    """Every run with its 10-session lift vs the universe and that test's BH q-value (patterns only)."""
    return reader.query("results", """
        SELECT r.run_id, r.kind, r.pattern_name, r.period, r.status, r.created_at, r.feature_build_id,
               l.n_dates, l.pattern_mean, l.baseline_mean, l.lift, l.t, q.q_value, r.invalid_reason
        FROM research_runs r
        LEFT JOIN pattern_lifts l ON l.run_id = r.run_id AND l.horizon = ?
        LEFT JOIN hypothesis_q q ON q.run_id = r.run_id AND q.label = 'lift_' || ?
        ORDER BY r.created_at DESC""", (LIFT_HORIZON, LIFT_HORIZON))


def _run_detail(con: duckdb.DuckDBPyConnection, run_id: str) -> dict:
    one = lambda sql: con.execute(sql, [run_id]).df()  # noqa: E731
    run = one("SELECT * FROM research_runs WHERE run_id = ?")
    out = {"run": run}
    if run.empty:
        return out
    out["definition"] = con.execute("""SELECT d.* FROM pattern_definitions d JOIN research_runs r
        ON d.name = r.pattern_name AND d.version = r.pattern_version WHERE r.run_id = ?""", [run_id]).df()
    out["lifts"] = one("SELECT * EXCLUDE (run_id) FROM pattern_lifts WHERE run_id = ? ORDER BY horizon")
    out["stats"] = one("SELECT * EXCLUDE (run_id) FROM pattern_stats WHERE run_id = ? ORDER BY horizon, segment")
    out["comparisons"] = one("SELECT * EXCLUDE (run_id) FROM pattern_comparisons WHERE run_id = ?")
    out["scan"] = one("""SELECT * EXCLUDE (run_id) FROM scan_stats WHERE run_id = ?
                         ORDER BY abs(ic_t) DESC NULLS LAST LIMIT 100""")
    out["tests"] = one("SELECT label, p_value, q_value, rk, m FROM hypothesis_q WHERE run_id = ? ORDER BY rk")
    return out


def run_detail(reader: Reader, run_id: str) -> ReadResult:
    return reader.read("results", ("run_detail", run_id), lambda con: _run_detail(con, run_id))


def hypothesis_log(reader: Reader) -> ReadResult:
    return reader.query("results", """SELECT q.run_id, q.label, q.period, q.kind, q.pattern_name, q.p_value,
                                             q.q_value, q.rk, q.m
                                      FROM hypothesis_q q ORDER BY q.q_value, q.p_value""")


def _registry(con: duckdb.DuckDBPyConnection) -> dict | None:
    if not _has(con, "hypotheses"):
        return None
    status = con.execute("""SELECT hypothesis_id, family, status, decision, title, reason, updated_at, result_doc,
                                   pattern, pattern_version, prereg_doc, statement
                            FROM hypothesis_status ORDER BY hypothesis_id""").df()
    history = con.execute("""SELECT hypothesis_id, seq, moved_at, status, decision, reason, run_id, prereg_doc,
                                    left(prereg_sha, 12) AS prereg_sha, result_doc
                             FROM hypothesis_transitions ORDER BY hypothesis_id, seq""").df()
    runs = con.execute("""
        WITH links AS (
            SELECT DISTINCT hypothesis_id, run_id FROM hypothesis_transitions WHERE run_id IS NOT NULL
            UNION
            SELECT h.hypothesis_id, r.run_id FROM hypotheses h JOIN research_runs r ON r.pattern_name = h.pattern)
        SELECT l.hypothesis_id, r.run_id, r.period, r.status, r.created_at, min(q.q_value) AS min_q, count(q.label) AS tests
        FROM links l JOIN research_runs r USING (run_id) LEFT JOIN hypothesis_q q USING (run_id)
        GROUP BY ALL ORDER BY l.hypothesis_id, r.created_at""").df()
    return {"status": status, "history": history, "runs": runs}


def registry(reader: Reader) -> ReadResult:
    """Research Registry: current state per hypothesis, append-only history, linked runs with min BH q."""
    return reader.read("results", "registry", _registry)


def _event_study(con: duckdb.DuckDBPyConnection) -> dict | None:
    if not _has(con, "event_study_stats"):
        return None
    run = con.execute("""SELECT run_id, created_at FROM research_runs WHERE kind = 'event_study' AND status = 'ok'
                         ORDER BY created_at DESC LIMIT 1""").fetchone()
    if run is None:
        return None
    return {"run_id": run[0], "created_at": run[1],
            "stats": con.execute("""SELECT event_type, horizon, segment, n_events, n_dates, mean, median, win_rate, std,
                                           p05, p95, mean_max_gain_5d, mean_max_loss_5d, lift, lift_t
                                    FROM event_study_stats WHERE run_id = ?
                                    ORDER BY event_type, horizon, segment""", [run[0]]).df()}


def event_study(reader: Reader) -> ReadResult:
    return reader.read("results", "event_study", _event_study)


# Backtest -------------------------------------------------------------------------------------

def backtest_runs(reader: Reader) -> ReadResult:
    """None when no backtest has ever run (the backtest tables are created by the first run)."""
    return reader.read("results", "backtest_runs", lambda con: _df_or_none(con, "backtest_metrics", """
        SELECT r.run_id, r.pattern_name AS strategy, r.period, r.status, r.created_at,
               max(m.value) FILTER (WHERE m.series = 'strategy' AND m.metric = 'cagr') AS cagr,
               max(m.value) FILTER (WHERE m.series = 'strategy' AND m.metric = 'sharpe') AS sharpe,
               max(m.value) FILTER (WHERE m.series = 'strategy' AND m.metric = 'max_drawdown') AS max_drawdown,
               max(m.value) FILTER (WHERE m.series = 'strategy' AND m.metric = 'exposure') AS exposure,
               max(m.value) FILTER (WHERE m.series = 'random_median' AND m.metric = 'cagr') AS random_median_cagr,
               max(m.value) FILTER (WHERE m.series = 'equal_weight' AND m.metric = 'cagr') AS equal_weight_cagr,
               max(m.value) FILTER (WHERE m.series = 'test' AND m.metric = 'share_of_random_runs_beaten')
                   AS beat_random_share,
               max(m.value) FILTER (WHERE m.series = 'test' AND m.metric = 'excess_vs_equal_weight_p') AS excess_p
        FROM research_runs r JOIN backtest_metrics m USING (run_id)
        WHERE r.kind = 'backtest'
        GROUP BY ALL ORDER BY r.created_at DESC"""))


def _backtest_detail(con: duckdb.DuckDBPyConnection, run_id: str) -> dict:
    one = lambda sql: con.execute(sql, [run_id]).df()  # noqa: E731
    return {
        "daily": one("SELECT * EXCLUDE (run_id) FROM backtest_daily WHERE run_id = ? ORDER BY date"),
        "metrics": one("SELECT series, metric, value FROM backtest_metrics WHERE run_id = ? ORDER BY series, metric"),
        "exits": one("""SELECT exit_reason, count(*) AS trades, sum(gross_pnl - costs) AS net_pnl,
                               avg(exit_price / entry_price - 1) AS avg_return
                        FROM backtest_trades WHERE run_id = ? GROUP BY 1 ORDER BY 2 DESC"""),
        "trades": one("""SELECT symbol, entry_date, exit_date, entry_price, exit_price, shares, gross_pnl, costs,
                                renewals, exit_reason
                         FROM backtest_trades WHERE run_id = ? ORDER BY entry_date DESC LIMIT 500"""),
        "params": one("SELECT params FROM research_runs WHERE run_id = ?"),
    }


def backtest_detail(reader: Reader, run_id: str) -> ReadResult:
    return reader.read("results", ("backtest_detail", run_id), lambda con: _backtest_detail(con, run_id))


def _paper(con: duckdb.DuckDBPyConnection) -> dict | None:
    if not _has(con, "paper_daily"):
        return None
    last = current_paper_date(con)
    if last is None:
        return {"run_date": None}
    return {
        "run_date": last,
        "daily": con.execute("SELECT * EXCLUDE (run_date) FROM paper_daily WHERE run_date = ? ORDER BY date",
                             [last]).df(),
        "positions": con.execute("""SELECT symbol, entry_date, shares, entry_price, last_price,
                                           last_price / entry_price - 1 AS pnl
                                    FROM paper_positions WHERE run_date = ? ORDER BY symbol""", [last]).df(),
    }


def paper(reader: Reader) -> ReadResult:
    return reader.read("results", "paper", _paper)


# Daily ----------------------------------------------------------------------------------------

def _daily(con: duckdb.DuckDBPyConnection) -> dict | None:
    if not _has(con, "daily_scans"):
        return None
    return {
        "scans": con.execute("SELECT * FROM daily_scans ORDER BY scan_date DESC").df(),
        "events": con.execute("""SELECT scan_date, pattern, symbol, is_forward, market_regime, exchange_now,
                                        close_raw, return_1d, volume_ratio_20, order_imbalance
                                 FROM daily_events ORDER BY scan_date DESC, pattern, symbol""").df(),
    }


def daily(reader: Reader) -> ReadResult:
    return reader.read("results", "daily_page", _daily)


def _interactions(con: duckdb.DuckDBPyConnection) -> dict | None:
    if not _has(con, "interaction_tests"):
        return None
    run = con.execute("""SELECT run_id FROM research_runs WHERE kind = 'interaction' AND status = 'ok'
                         ORDER BY created_at DESC LIMIT 1""").fetchone()
    if run is None:
        return None
    from quant_research.interaction_report import is_candidate
    tests = con.execute("""
        SELECT t.*, q.q_value FROM interaction_tests t
        LEFT JOIN hypothesis_q q ON q.run_id = t.run_id AND q.label = 'ix_' || t.factor || '_' || t.dimension
        WHERE t.run_id = ? ORDER BY t.factor, t.dimension""", [run[0]]).df()
    tests["candidate"] = tests.apply(is_candidate, axis=1)
    stats = con.execute("SELECT * EXCLUDE (run_id) FROM interaction_stats WHERE run_id = ?", [run[0]]).df()
    return {"run_id": run[0], "tests": tests, "stats": stats}


def interactions(reader: Reader) -> ReadResult:
    """Latest valid interaction scan: tests with BH q-values and the declared candidate rule; per-state ICs."""
    return reader.read("results", "interactions", _interactions)


current_paper_date = paper_state.current_paper_date      # moved to quant_research (Risk Engine uses it)

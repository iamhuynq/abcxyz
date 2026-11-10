"""Cross-stock hypothesis tests (doc 7-10), logged in results.duckdb like every other test.

- Lead-lag persistence (family level, doc 7 + "is the relation stable?"): rank directed pairs by their
  lag correlation in year Y; does the top decile still beat all pairs in year Y + 1?
- H1 industry leaders -> followers (doc 8): cross-sectional decile scan of the leaders' excess return
  today on the followers' future execution returns (Phase 3 scan engine).
- H2 large caps -> small caps: time-series slope of the small-cap basket's future excess return on the
  large-cap basket's return today, Newey-West.
- H3 strong leader day, follower not moved yet (doc 8): a pattern on the cross-stock view.
- Cointegration long leg (doc 10): a pattern on the cheap leg of a spread event.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats as sps

from quant_research.cross import group_tests
from quant_research.cross.params import CrossParams
from quant_research.engine import run_pattern, run_scan
from quant_research.patterns import Pattern
from quant_research.results import ResearchParams, ResultsStore, check_period
from quant_research.stats import mean_test

SCHEMA = """
CREATE TABLE IF NOT EXISTS cross_stats (
    run_id VARCHAR NOT NULL, test VARCHAR NOT NULL, horizon VARCHAR NOT NULL, n BIGINT, estimate DOUBLE,
    se DOUBLE, t DOUBLE, p DOUBLE, detail VARCHAR, PRIMARY KEY (run_id, test, horizon));
"""
VIEW = "cross_target"
TEST_YEARS = {"research": (1900, 2023), "validation": (2024, 2025)}     # lead-lag test years per period
TOP_SHARE = 0.10

CROSS_LIBRARY: dict[str, Pattern] = {p.name: p for p in (
    Pattern("cross_h3_leader_jump_follower_flat",
            "NOT cx_is_leader AND leader_excess_1d >= 0.05 AND own_excess_1d <= 0.01",
            "H3: when the industry leaders jump (+5% vs VNINDEX) and a follower has not moved yet, the follower "
            "catches up over the next sessions", "8"),
    Pattern("cross_g4_laggard_in_strong_industry", "grp_tercile = 3 AND own_bottom_half",
            "G4: a stock in the bottom half of its industry over 1 month, while the industry is in the top "
            "tercile of industries over 1 month, catches up (doc 8, leaders and laggards)", "8, 11"),
    Pattern("cross_coint_long_leg", "coint_long",
            "Cointegration: the cheap leg of a same-industry pair whose spread just moved beyond 2 sigma "
            "outperforms (long leg only; no short selling)", "10"),
)}


def attach(store: ResultsStore, cross_path: Path) -> None:
    """Attach cross.duckdb read-only and define the cross-stock view over feature_target."""
    if not cross_path.exists():
        raise FileNotFoundError(f"No cross database at {cross_path}; run `quant cross build`")
    store.con.execute(SCHEMA)
    store.con.execute(f"ATTACH '{_sql_path(cross_path)}' AS cx (READ_ONLY)")
    store.con.execute(f"""CREATE OR REPLACE TEMP VIEW {VIEW} AS
        SELECT f.*, coalesce(c.is_leader, false) AS cx_is_leader, c.leader_excess_1d, c.own_excess_1d,
               c.large_ret_1d, coalesce(c.in_cross_universe, false) AS in_cross_universe,
               e.symbol IS NOT NULL AS coint_long,
               g.grp_tercile, coalesce(g.own_bottom_half, false) AS own_bottom_half
        FROM rs.feature_target f
        LEFT JOIN cx.cross_features c USING (symbol, date)
        LEFT JOIN (SELECT symbol, date, grp_tercile, own_bottom_half FROM cx.group_features) g USING (symbol, date)
        LEFT JOIN (SELECT DISTINCT symbol, date FROM cx.coint_events) e USING (symbol, date)""")


def _sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _cross_build(store: ResultsStore) -> dict:
    row = store.con.execute("SELECT build_id, params, code_hash FROM cx.cross_builds").fetchone()
    return {"cross_build_id": row[0], "cross_params": json.loads(row[1]), "cross_code_hash": row[2]}


def _record(store: ResultsStore, run_id: str, test: str, horizon: str, n: int, est, se, t, p, period: str,
            now: datetime, detail: dict | None = None, log: bool = True) -> None:
    store.con.execute("INSERT INTO cross_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      [run_id, test, horizon, n, est, se, t, p, json.dumps(detail or {})])
    if log:
        store.log_hypothesis(run_id, f"{test}_{horizon}", period, p, now)


PAIR_SOURCES = {   # name: (table, fixed filter, version)
    "leadlag_persistence": ("cx.leadlag_pairs", "TRUE", "v2"),
    "group_leadlag_persistence": ("cx.group_leadlag_pairs", "level = 'l2' AND lead <> follow", "v1"),
}


def run_leadlag_persistence(store: ResultsStore, period: str, params: ResearchParams, now: datetime,
                            cross: CrossParams, final: bool = False, name: str = "leadlag_persistence") -> str:
    """Per lag: series over test years of (top-decile pairs' test corr - all pairs' test corr); t-test."""
    check_period(period, final)
    if period not in TEST_YEARS:
        raise ValueError("lead-lag persistence is defined per calendar year: research or validation only")
    lo, hi = TEST_YEARS[period]
    table, where, version = PAIR_SOURCES[name]
    prefix = "group_" if name.startswith("group_") else ""
    run_id = f"{now:%Y%m%dT%H%M%S}-cross-{name}-{period}"
    store.start_run(run_id, "cross", period, params, now, (name, version))
    for lag in cross.lags:
        for series in ("cc", "oc"):
            rows = store.con.execute(f"""
                WITH ranked AS (    -- the top decile is chosen from formation data only (all pairs)
                    SELECT form_year, corr_form, corr_test_{series} AS y,
                           percent_rank() OVER (PARTITION BY form_year ORDER BY corr_form DESC) AS pr
                    FROM {table} WHERE {where} AND lag = ? AND form_year + 1 BETWEEN ? AND ?),
                p AS (SELECT * FROM ranked WHERE y IS NOT NULL AND isfinite(y))   -- measurable in Y + 1
                SELECT form_year + 1, avg(y) FILTER (WHERE pr < ?) - avg(y), avg(y) FILTER (WHERE pr < ?), count(*)
                FROM p GROUP BY 1 HAVING count(*) FILTER (WHERE pr < ?) > 0 ORDER BY 1""",
                [lag, lo, hi, TOP_SHARE, TOP_SHARE, TOP_SHARE]).fetchall()
            d = np.array([r[1] for r in rows], dtype=float)
            se, t, p, _, _ = mean_test(d, 0)
            detail = {"years": [r[0] for r in rows], "excess": [round(x, 5) for x in d.tolist()],
                      "top_mean": [round(r[2], 5) for r in rows], "pairs": [r[3] for r in rows]}
            _record(store, run_id, f"{prefix}leadlag_persist_{series}", f"lag{lag}", len(d),
                    float(d.mean()) if len(d) else None, se, t, p, period, now, detail)
    store.finish_run(run_id)
    return run_id


def hac_ols(y: np.ndarray, x: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS with an intercept; (coefficients, Newey-West standard errors), Bartlett kernel."""
    X = np.column_stack([np.ones(len(y)), x])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    g = X * (y - X @ beta)[:, None]
    s = g.T @ g
    for k in range(1, min(lag, len(y) - 1) + 1):
        w = 1 - k / (lag + 1)
        gk = g[k:].T @ g[:-k]
        s += w * (gk + gk.T)
    cov = xtx_inv @ s @ xtx_inv
    return beta, np.sqrt(np.maximum(np.diag(cov), 0.0))


def run_h2_large_to_small(store: ResultsStore, period: str, params: ResearchParams, now: datetime,
                          final: bool = False) -> str:
    """Small-cap basket's future excess return on the large-cap basket's EXCESS return today, controlling
    for the market return today (otherwise market autocorrelation x small-cap beta looks like a lead-lag)."""
    check_period(period, final)
    run_id = f"{now:%Y%m%dT%H%M%S}-cross-h2_large_to_small-{period}"
    store.start_run(run_id, "cross", period, params, now, ("h2_large_to_small", "v2"))
    for h in params.horizons:
        y = f"fwd_excess_exec_{h}d"
        rows = store.con.execute(f"""
            SELECT v.date, any_value(v.large_ret_1d) - any_value(m.mkt_ret_1d) AS x, any_value(m.mkt_ret_1d) AS mkt,
                   avg(v.{y}) AS y
            FROM {VIEW} v JOIN rs.market_daily m USING (date)
            WHERE {params.sample_sql(y)} AND v.period = ? AND NOT v.in_cross_universe AND v.large_ret_1d IS NOT NULL
              AND m.mkt_ret_1d IS NOT NULL AND v.{y} IS NOT NULL
            GROUP BY v.date HAVING count(*) >= ? ORDER BY v.date""", [period, params.min_stocks_per_date]).fetchall()
        arr = np.array([r[1:] for r in rows], dtype=float).reshape(-1, 3)
        est = se = t = p = None
        detail: dict = {}
        if len(arr) > 10 and arr[:, 0].var() > 0:
            beta, ses = hac_ols(arr[:, 2], arr[:, :2], h - 1)
            est, se = float(beta[1]), float(ses[1])
            t = est / se if se else None
            p = float(2 * sps.t.sf(abs(t), len(arr) - 3)) if t is not None else None
            detail = {"market_coef": float(beta[2]), "market_t": float(beta[2] / ses[2]) if ses[2] else None,
                      "mean_small_basket": float(arr[:, 2].mean())}
        _record(store, run_id, "h2_slope", f"exec_excess_{h}d", len(arr), est, se, t, p, period, now, detail)
    store.finish_run(run_id)
    return run_id


# test key: (run kind, run name) - the run name is also how an earlier run is recognised
TEST_RUNS = {
    "leadlag": ("cross", "leadlag_persistence"),
    "h1": ("cross", "h1_leaders_to_followers"),
    "h2": ("cross", "h2_large_to_small"),
    "h3": ("pattern", "cross_h3_leader_jump_follower_flat"),
    "coint": ("pattern", "cross_coint_long_leg"),
    "g_leadlag": ("cross", "group_leadlag_persistence"),
    "g_momentum": ("cross", "group_momentum"),
    "g_reversal": ("cross", "group_reversal"),
    "g_laggard": ("pattern", "cross_g4_laggard_in_strong_industry"),
}
TESTS = tuple(TEST_RUNS)


def already_run(store: ResultsStore, test: str, period: str) -> str | None:
    """The valid earlier run of this test and period, if any (re-running would log the same tests twice)."""
    kind, name = TEST_RUNS[test]
    row = store.con.execute("""SELECT run_id FROM research_runs WHERE status = 'ok' AND period = ?
                               AND run_id LIKE ? ORDER BY created_at DESC LIMIT 1""",
                            [period, f"%-{kind}-{name}-{period}"]).fetchone()
    return row[0] if row else None


def run_all(store: ResultsStore, period: str, params: ResearchParams, now: datetime, cross: CrossParams,
            final: bool = False, tests: tuple[str, ...] = TESTS, log=print) -> list[str]:
    """Runs the selected tests; a test with a valid earlier run for this period is skipped (invalidate it
    with a reason first to run it again)."""
    unknown = set(tests) - set(TESTS)
    if unknown:
        raise ValueError(f"unknown tests {sorted(unknown)}; choose from {', '.join(TESTS)}")
    params = ResearchParams(**{**asdict(params), "extra": {**params.extra, **_cross_build(store)}})
    runs = []
    for test in tests:
        earlier = already_run(store, test, period)
        if earlier:
            log(f"skip {test}: already run as {earlier} (invalidate it to run again)")
            continue
        if test in ("leadlag", "g_leadlag") and period not in TEST_YEARS:
            log(f"skip {test}: defined per calendar year (research or validation only)")
            continue
        runs.append(_run_one(store, test, period, params, now, cross, final))
    return runs


def _run_one(store: ResultsStore, test: str, period: str, params: ResearchParams, now: datetime,
             cross: CrossParams, final: bool) -> str:
    kind, name = TEST_RUNS[test]
    if test in ("leadlag", "g_leadlag"):
        return run_leadlag_persistence(store, period, params, now, cross, final, name=name)
    if test == "h1":
        return run_scan(store, period, params, now, final, features=("leader_excess_1d",), table=VIEW,
                        row_filter="NOT cx_is_leader AND leader_excess_1d IS NOT NULL", name=name, kind="cross")
    if test == "h2":
        return run_h2_large_to_small(store, period, params, now, final)
    if test == "g_momentum":
        return group_tests.run_group_momentum(store, period, params, now, _record, final)
    if test == "g_reversal":
        return group_tests.run_group_reversal(store, period, params, now, _record, final)
    return run_pattern(store, CROSS_LIBRARY[name], period, params, now, final, library=CROSS_LIBRARY, table=VIEW)

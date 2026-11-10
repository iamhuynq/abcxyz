"""Daily scanner (doc 15, 24, 25) and forward paper trading.

- Scans only `rs.stock_features` (no target columns) with the point-in-time universe, one date at a time.
- Catch-up: every forward session not yet scanned is scanned once; the latest session is always scanned
  for display (flagged is_forward = false if it is before FORWARD_START).
- Completeness gate: a session is scanned only if its panel row count is >= COMPLETENESS of the
  previous session's (a partial `fireant update` must not produce a partial cross-section).
- Paper trading re-simulates the frozen strategy over the forward window with the Phase 6 engine.
- Only final sessions (fireant_crawler.sessions.final_session) are scanned or traded: a session still in
  progress, if it ever reaches the warehouse, is ignored until it is final.
"""

from dataclasses import dataclass
from datetime import date, datetime

import duckdb
import numpy as np

from fireant_crawler.sessions import final_session
from quant_research.backtest.data import UniverseRule, load_market
from quant_research.backtest.engine import Strategy, random_selector, run, signal_selector
from quant_research.backtest.metrics import equal_weight_curve
from quant_research.events import DAILY_PREFIX, catalog_version
from quant_research.patterns import LIBRARY, Pattern
from quant_research.results import ResultsStore

FORWARD_START = date(2026, 10, 3)
COMPLETENESS = 0.95
AVOID_PATTERN = "drop3_volume2_sellers"
FROZEN_STRATEGY = Strategy(hold_sessions=10, max_positions=20, renew=True, initial_equity=1e9)  # version 72c851c7
N_RANDOM = 20            # pre-registered (preregistration-backtest-2026-10-04.md: seeds 0-19); never change
N_RANDOM_WIDE = 200      # diagnostic only (review of 2026-11-10): not the pre-registered criterion
# Validation decisions live in the Research Registry (quant_research/registry.py). Reports show them
# verbatim; they never recompute a pass/fail from statistics (that once mislabelled C1 as passed).

EVENT_COLUMNS = ("market_regime", "exchange_now", "close_raw", "return_1d", "volume_ratio_20", "order_imbalance")

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_scans (
    scan_date DATE PRIMARY KEY, status VARCHAR NOT NULL, coverage DOUBLE, is_forward BOOLEAN NOT NULL,
    n_events INTEGER, feature_build_id VARCHAR, scanned_at TIMESTAMPTZ NOT NULL);
CREATE TABLE IF NOT EXISTS daily_events (
    scan_date DATE NOT NULL, pattern VARCHAR NOT NULL, version VARCHAR NOT NULL, symbol VARCHAR NOT NULL,
    is_forward BOOLEAN NOT NULL, market_regime VARCHAR, exchange_now VARCHAR, close_raw DOUBLE, return_1d DOUBLE,
    volume_ratio_20 DOUBLE, order_imbalance DOUBLE, PRIMARY KEY (scan_date, pattern, symbol));
CREATE TABLE IF NOT EXISTS paper_daily (
    run_date DATE NOT NULL, date DATE NOT NULL, equity DOUBLE, random_p05 DOUBLE, random_median DOUBLE,
    random_p95 DOUBLE, equal_weight DOUBLE, vnindex DOUBLE, n_positions INTEGER, PRIMARY KEY (run_date, date));
-- Added 2026-10-09 (execution audit): the same strategy marked at last traded closes (information only).
ALTER TABLE paper_daily ADD COLUMN IF NOT EXISTS equity_traded_mark DOUBLE;
-- The run_date of the paper record that is current (NULL: no complete forward session yet).
-- Added 2026-11-10 (review): 200 random controls (seeds 0-199), diagnostic only; the pre-registered
-- criterion stays on the 20 controls above. exceedance p = (1 + controls >= strategy) / (1 + controls).
ALTER TABLE paper_daily ADD COLUMN IF NOT EXISTS wide_random_median DOUBLE;
ALTER TABLE paper_daily ADD COLUMN IF NOT EXISTS wide_random_p95 DOUBLE;
ALTER TABLE paper_daily ADD COLUMN IF NOT EXISTS wide_exceedance_p DOUBLE;
CREATE TABLE IF NOT EXISTS paper_state (id INTEGER PRIMARY KEY, paper_date DATE, updated_at TIMESTAMPTZ);
CREATE TABLE IF NOT EXISTS paper_positions (
    run_date DATE NOT NULL, symbol VARCHAR NOT NULL, entry_date DATE, shares DOUBLE, entry_price DOUBLE,
    last_price DOUBLE, PRIMARY KEY (run_date, symbol));
"""


@dataclass
class DailyResult:
    latest_date: date | None
    scanned: list[date]
    incomplete: list[tuple[date, float]]
    paper_sessions: int
    paper_date: date | None = None       # latest complete session: the paper portfolio never uses a partial one


def coverage(con: duckdb.DuckDBPyConnection, d: date) -> float:
    """Raw warehouse rows on d divided by rows on the previous calendar session (see 02b_source_coverage.sql)."""
    prev = con.execute("SELECT max(date) FROM rs.market_daily WHERE date < ?", [d]).fetchone()[0]
    if prev is None:
        return 1.0
    rows = dict(con.execute("SELECT date, n_rows FROM rs.source_coverage WHERE date IN (?, ?)", [d, prev]).fetchall())
    return rows.get(d, 0) / rows[prev] if rows.get(prev) else 1.0


def latest_final(con: duckdb.DuckDBPyConnection, final: date) -> date | None:
    return con.execute("SELECT max(date) FROM rs.market_daily WHERE date <= ?", [final]).fetchone()[0]


def sessions_to_scan(con: duckdb.DuckDBPyConnection, final: date, rescan: date | None = None) -> list[date]:
    if rescan:
        if rescan > final:
            raise ValueError(f"{rescan} is not final yet (last final session: {final})")
        return [rescan]
    latest = latest_final(con, final)
    pending = [r[0] for r in con.execute("""
        SELECT date FROM rs.market_daily WHERE date >= ? AND date <= ?
          AND date NOT IN (SELECT scan_date FROM daily_scans WHERE status = 'scanned')
        ORDER BY date""", [FORWARD_START, final]).fetchall()]
    done_latest = con.execute("SELECT count(*) FROM daily_scans WHERE scan_date = ? AND status = 'scanned'",
                              [latest]).fetchone()[0]
    if latest is not None and latest not in pending and not done_latest:
        pending.append(latest)
    return pending


def scan_date(store: ResultsStore, d: date, now: datetime, rule: UniverseRule = UniverseRule(),
              library: dict[str, Pattern] = LIBRARY) -> tuple[str, float, int]:
    con = store.con
    cov = coverage(con, d)
    is_forward = d >= FORWARD_START
    build_id = store.feature_build_id()
    if cov < COMPLETENESS:
        con.execute("INSERT OR REPLACE INTO daily_scans VALUES (?, 'incomplete', ?, ?, 0, ?, ?)",
                    [d, cov, is_forward, build_id, now])
        return "incomplete", cov, 0
    con.execute("DELETE FROM daily_events WHERE scan_date = ?", [d])
    n = 0
    cols = ", ".join(EVENT_COLUMNS)
    for p in library.values():
        src = p.source_sql(rule.sql(), rule.min_stocks_per_date, table="rs.stock_features", row_filter="date = ?")
        cur = con.execute(f"""INSERT INTO daily_events
            SELECT ?, ?, ?, symbol, ?, {cols} FROM ({src})""", [d, p.name, p.version, is_forward, d])
        n += cur.fetchone()[0]
    n += _scan_catalog_events(con, d, is_forward, rule, cols)
    con.execute("INSERT OR REPLACE INTO daily_scans VALUES (?, 'scanned', ?, ?, ?, ?, ?)",
                [d, cov, is_forward, n, build_id, now])
    return "scanned", cov, n


def _scan_catalog_events(con: duckdb.DuckDBPyConnection, d: date, is_forward: bool, rule: UniverseRule,
                         cols: str) -> int:
    """Copy the scan date's catalog events (rs.stock_events, Phase 5) into daily_events as EVENT_<TYPE>."""
    has = con.execute("""SELECT count(*) FROM duckdb_tables()
                         WHERE database_name = 'rs' AND table_name = 'stock_events'""").fetchone()[0]
    if not has:
        return 0
    prefixed = ", ".join(f"f.{c}" for c in cols.split(", "))
    cur = con.execute(f"""INSERT INTO daily_events
        SELECT ?, '{DAILY_PREFIX}' || e.event_type, ?, f.symbol, ?, {prefixed}
        FROM rs.stock_features f JOIN rs.stock_events e USING (symbol, date)
        WHERE {rule.sql('f.')} AND f.date = ?""", [d, catalog_version(), is_forward, d])
    return cur.fetchone()[0]


def run_paper(store: ResultsStore, run_date: date, strategy: Strategy = FROZEN_STRATEGY,
              rule: UniverseRule = UniverseRule(), n_random: int = N_RANDOM,
              n_wide: int = N_RANDOM_WIDE) -> int:
    """Re-simulate the frozen strategy over the forward window up to run_date; returns the number of sessions."""
    m = load_market(store.con, FORWARD_START, run_date, rule)
    if len(m.dates) < 2:                             # nothing to simulate: leave no rows behind for this run_date
        store.con.execute("DELETE FROM paper_daily WHERE run_date = ?", [run_date])
        store.con.execute("DELETE FROM paper_positions WHERE run_date = ?", [run_date])
        return len(m.dates)
    res = run(m, strategy, signal_selector, mark="legacy")        # pre-registered measurement
    wide = np.array([run(m, strategy, random_selector(seed), mark="legacy").equity
                     for seed in range(max(n_random, n_wide))])
    rnd = wide[:n_random]                            # pre-registered: seeds 0-19, the same runs as before
    wide = wide[:n_wide] if n_wide else None
    traded = run(m, strategy, signal_selector, mark="traded").equity      # information: last traded closes
    ew = equal_weight_curve(m.close, m.universe, strategy.initial_equity)
    vn = strategy.initial_equity * m.index_close / m.index_close[0]
    p05, med, p95 = np.percentile(rnd, [5, 50, 95], axis=0)
    if wide is not None and len(wide):
        w_med, w_p95 = np.percentile(wide, [50, 95], axis=0)
        w_exc = (1 + (wide >= res.equity).sum(axis=0)) / (1 + len(wide))
    else:
        w_med = w_p95 = w_exc = np.full(len(m.dates), np.nan)
    con = store.con
    con.execute("DELETE FROM paper_daily WHERE run_date = ?", [run_date])
    con.execute("DELETE FROM paper_positions WHERE run_date = ?", [run_date])
    con.executemany("""INSERT INTO paper_daily (run_date, date, equity, random_p05, random_median, random_p95,
                       equal_weight, vnindex, n_positions, equity_traded_mark, wide_random_median, wide_random_p95,
                       wide_exceedance_p) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", [
        (run_date, d.item(), float(res.equity[i]), float(p05[i]), float(med[i]), float(p95[i]), float(ew[i]),
         float(vn[i]), int(res.n_positions[i]), float(traded[i]), _num(w_med[i]), _num(w_p95[i]), _num(w_exc[i]))
        for i, d in enumerate(m.dates)])
    open_rows = [(run_date, t.symbol, m.dates[t.entry_index].item(), t.shares, t.entry_price, t.exit_price)
                 for t in res.trades if t.exit_reason == "open_at_end"]
    if open_rows:                                    # no open position (all cash) is valid: nothing to insert
        con.executemany("INSERT INTO paper_positions VALUES (?, ?, ?, ?, ?, ?)", open_rows)
    return len(m.dates)


def _num(v) -> float | None:
    return None if v is None or not np.isfinite(v) else float(v)


def run_daily(store: ResultsStore, now: datetime, rescan: date | None = None,
              library: dict[str, Pattern] = LIBRARY, n_random: int = N_RANDOM,
              n_wide: int = N_RANDOM_WIDE,
              rule: UniverseRule = UniverseRule()) -> DailyResult:
    store.con.execute(SCHEMA)
    final = final_session(now)
    latest = latest_final(store.con, final)
    scanned, incomplete = [], []
    for d in sessions_to_scan(store.con, final, rescan):
        status, cov, _ = scan_date(store, d, now, rule=rule, library=library)
        (scanned.append(d) if status == "scanned" else incomplete.append((d, cov)))
    # Completeness gate for the paper portfolio too: it is re-simulated from FORWARD_START, so it may only run up
    # to the end of the unbroken run of complete (status 'scanned') forward sessions. It stops before the first
    # forward session that is not complete, even if a later one is, and catches up once that session is complete.
    # A scanned session is only re-scanned by `quant daily --date`; the paper record is rebuilt on every run.
    paper_date = store.con.execute("""
        WITH f AS (SELECT date FROM rs.market_daily WHERE date >= ? AND date <= ?),
        gap AS (SELECT min(date) AS d FROM f
                WHERE date NOT IN (SELECT scan_date FROM daily_scans WHERE status = 'scanned'))
        SELECT max(s.scan_date) FROM daily_scans s, gap
        WHERE s.status = 'scanned' AND s.scan_date >= ? AND s.scan_date <= ? AND (gap.d IS NULL OR s.scan_date < gap.d)""",
        [FORWARD_START, latest, FORWARD_START, latest]).fetchone()[0] if latest else None
    sessions = run_paper(store, paper_date, rule=rule, n_random=n_random, n_wide=n_wide) if paper_date else 0
    # The current paper record: readers use this date, never max(run_date), so an older re-simulation that is no
    # longer valid (or none at all) is never shown as current. Earlier run_date rows stay as history.
    # Only a record that was actually simulated (2+ forward sessions) becomes current; otherwise there is none.
    store.con.execute("INSERT OR REPLACE INTO paper_state VALUES (1, ?, ?)", [paper_date if sessions >= 2 else None, now])
    return DailyResult(latest, scanned, incomplete, sessions, paper_date)

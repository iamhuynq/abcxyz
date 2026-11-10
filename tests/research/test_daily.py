"""Daily scanner, completeness gate, catch-up, paper portfolio and report."""

from datetime import UTC, date, datetime

import duckdb
import numpy as np
import pytest

from fireant_crawler.store.warehouse import Warehouse
from quant_research.backtest.data import UniverseRule, load_market
from quant_research.backtest.engine import random_selector, run, signal_selector
from quant_research.build import build
from quant_research.daily import FORWARD_START, FROZEN_STRATEGY, run_daily, run_paper
from quant_research.daily_report import render_daily, write_reports
from quant_research.patterns import Pattern
from quant_research.results import ResultsStore
from tests.research.synthetic import build_synthetic_warehouse, calendar

NOW = datetime(2027, 3, 1, 19, 0, tzinfo=UTC)   # after the synthetic calendar: every session is final
RULE = UniverseRule(min_adv_value=0, min_session_index=0, min_stocks_per_date=2, signal_decile=3)
START = date(2026, 1, 5)
FORWARD_DAYS = [d for d in calendar(START) if d >= FORWARD_START]


def make(tmp_path, start=START, drop_last_for=None):
    wh, rs = tmp_path / "w.duckdb", tmp_path / "r.duckdb"
    for p in (wh, rs):
        p.unlink(missing_ok=True)
    build_synthetic_warehouse(wh, start=start)
    if drop_last_for:
        with Warehouse(wh) as w:
            last = w.connection.execute("SELECT max(date) FROM quotes_daily").fetchone()[0]
            w.connection.execute("DELETE FROM quotes_daily WHERE date = ? AND symbol = ?", [last, drop_last_for])
    build(wh, rs, NOW)
    return rs


def test_catch_up_scans_every_forward_session_once(tmp_path):
    rs = make(tmp_path)
    with ResultsStore(tmp_path / "results.duckdb", rs) as store:
        first = run_daily(store, NOW, rule=RULE, n_random=2)
        second = run_daily(store, NOW, rule=RULE, n_random=2)
        n_scans = store.con.execute("SELECT count(*) FROM daily_scans WHERE is_forward").fetchone()[0]
    assert first.scanned == FORWARD_DAYS and first.incomplete == []
    assert second.scanned == [] and n_scans == len(FORWARD_DAYS)


def test_events_equal_a_direct_query_on_features_only(tmp_path):
    rs = make(tmp_path)
    lib = {"down": Pattern("down", "return_1d < 0", "test: fires on most sessions")}
    with ResultsStore(tmp_path / "results.duckdb", rs) as store:
        run_daily(store, NOW, rule=RULE, n_random=2, library=lib)
        got = store.con.execute("SELECT scan_date, symbol FROM daily_events WHERE pattern = 'down' ORDER BY 1, 2").fetchall()
        want = store.con.execute(f"""SELECT date, symbol FROM rs.stock_features
            WHERE date >= ? AND {RULE.sql()} AND return_1d < 0 ORDER BY 1, 2""", [FORWARD_START]).fetchall()
    assert got == want and len(got) > 20


def test_completeness_gate_skips_a_partial_session_until_complete(tmp_path):
    rs = make(tmp_path, drop_last_for="STK")                         # 4 of 5 raw rows on the last day
    with ResultsStore(tmp_path / "results.duckdb", rs) as store:
        r = run_daily(store, NOW, rule=RULE, n_random=2)
    assert r.incomplete and r.incomplete[0][0] == FORWARD_DAYS[-1] and r.incomplete[0][1] < 0.95
    assert FORWARD_DAYS[-1] not in r.scanned

    rs = make(tmp_path)                                               # the next update completes the day
    with ResultsStore(tmp_path / "results.duckdb", rs) as store:
        r = run_daily(store, NOW, rule=RULE, n_random=2)
        status = store.con.execute("SELECT status FROM daily_scans WHERE scan_date = ?", [FORWARD_DAYS[-1]]).fetchone()[0]
    assert r.scanned == [FORWARD_DAYS[-1]] and status == "scanned"


def test_delisting_and_no_trade_days_do_not_trip_the_gate(tmp_path):
    rs = make(tmp_path)                                               # DEL stops trading mid-sample
    with ResultsStore(tmp_path / "results.duckdb", rs) as store:
        r = run_daily(store, NOW, rule=RULE, n_random=2)
    assert r.incomplete == []


def test_paper_portfolio_is_a_deterministic_engine_run_on_the_forward_window(tmp_path):
    rs = make(tmp_path)
    with ResultsStore(tmp_path / "results.duckdb", rs) as store:
        run_daily(store, NOW, rule=RULE, n_random=2)
        eq = [r[0] for r in store.con.execute("SELECT equity FROM paper_daily ORDER BY date").fetchall()]
        run_paper(store, FORWARD_DAYS[-1], rule=RULE, n_random=2)
        eq2 = [r[0] for r in store.con.execute("SELECT equity FROM paper_daily ORDER BY date").fetchall()]
        direct = run(load_market(store.con, FORWARD_START, date(2099, 12, 31), RULE), FROZEN_STRATEGY, signal_selector)
    assert len(eq) == len(FORWARD_DAYS) and eq == eq2
    np.testing.assert_allclose(eq, direct.equity)


def test_wide_controls_are_diagnostic_and_leave_the_preregistered_columns_unchanged(tmp_path, monkeypatch):
    """The synthetic forward market is too small for random controls to trade, so each control's curve is
    replaced by one that encodes its seed: the pre-registered band must come from seeds 0..n_random-1."""
    import types

    import quant_research.daily as daily
    real_run = daily.run

    def fake_run(m, s, selector=None, *args, **kwargs):
        if isinstance(selector, tuple):
            return types.SimpleNamespace(equity=np.full(len(m.dates), 1e9 * (1 + 0.01 * selector[1])))
        return real_run(m, s, selector, *args, **kwargs)
    monkeypatch.setattr(daily, "random_selector", lambda seed: ("seed", seed))
    monkeypatch.setattr(daily, "run", fake_run)
    rs = make(tmp_path)
    cols = "random_p05, random_median, random_p95, wide_random_median, wide_exceedance_p, equity"
    with ResultsStore(tmp_path / "results.duckdb", rs) as store:
        run_daily(store, NOW, rule=RULE, n_random=4, n_wide=0)
        narrow = store.con.execute(f"SELECT {cols} FROM paper_daily ORDER BY date").fetchall()
        run_paper(store, FORWARD_DAYS[-1], rule=RULE, n_random=4, n_wide=30)
        wide = store.con.execute(f"SELECT {cols} FROM paper_daily ORDER BY date").fetchall()
    seeds4, seeds30 = 1e9 * (1 + 0.01 * np.arange(4)), 1e9 * (1 + 0.01 * np.arange(30))
    assert [r[:3] for r in narrow] == [r[:3] for r in wide]           # the same pre-registered band
    assert wide[-1][:3] == pytest.approx(tuple(np.percentile(seeds4, [5, 50, 95])))
    assert all(r[3] is None and r[4] is None for r in narrow)
    assert wide[-1][3] == pytest.approx(np.median(seeds30))
    assert wide[-1][4] == pytest.approx((1 + (seeds30 >= wide[-1][5]).sum()) / 31)


def test_report_renders_before_and_after_the_forward_start(tmp_path):
    rs = make(tmp_path)
    with ResultsStore(tmp_path / "results.duckdb", rs) as store:
        report = render_daily(store, run_daily(store, NOW, rule=RULE, n_random=2), NOW)
    assert "## Avoid list" in report and "## Forward scoreboard" in report and "Strategy:" in report

    old = tmp_path / "old"
    old.mkdir()
    rs_old = make(old, start=date(2023, 1, 2))                       # data ends before FORWARD_START
    with ResultsStore(old / "results.duckdb", rs_old) as store:
        result = run_daily(store, NOW, rule=RULE, n_random=2)
        report = render_daily(store, result, NOW)
        flags = store.con.execute("SELECT DISTINCT is_forward FROM daily_scans").fetchall()
    assert "not started yet" in report and "Not started" in report and flags == [(False,)]


def test_report_shows_recorded_validation_decisions_not_recomputed_ones(tmp_path):
    rs = make(tmp_path)
    with ResultsStore(tmp_path / "results.duckdb", rs) as store:
        report = render_daily(store, run_daily(store, NOW, rule=RULE, n_random=2), NOW)
    row = next(line for line in report.splitlines() if line.startswith("| order_imbalance_d10 |"))
    assert "**FAIL**" in row and "after costs" in row
    assert "**PASS**" in next(line for line in report.splitlines() if line.startswith("| drop3_volume2_sellers |"))
    assert "not validated" in next(line for line in report.splitlines() if line.startswith("| drop3 |"))


def test_a_session_in_progress_is_neither_scanned_nor_traded(tmp_path):
    """Even if the warehouse holds a row for today's session, quant daily waits until it is final."""
    from zoneinfo import ZoneInfo
    vn = ZoneInfo("Asia/Ho_Chi_Minh")
    rs = make(tmp_path)
    monday = next(d for d in FORWARD_DAYS[3:] if d.weekday() == 0)
    before = [d for d in FORWARD_DAYS if d < monday]
    with ResultsStore(tmp_path / "results.duckdb", rs) as store:
        morning = run_daily(store, datetime(monday.year, monday.month, monday.day, 10, 0, tzinfo=vn),
                            rule=RULE, n_random=2)
        last_paper = store.con.execute("SELECT max(date) FROM paper_daily").fetchone()[0]
        with pytest.raises(ValueError, match="not final"):
            run_daily(store, datetime(monday.year, monday.month, monday.day, 10, 0, tzinfo=vn),
                      rescan=monday, rule=RULE, n_random=2)
        evening = run_daily(store, datetime(monday.year, monday.month, monday.day, 18, 30, tzinfo=vn),
                            rule=RULE, n_random=2)
    assert morning.scanned == before and morning.latest_date == before[-1] and last_paper == before[-1]
    assert evening.scanned == [monday] and evening.latest_date == monday


def test_catch_up_run_writes_one_report_per_scanned_session(tmp_path):
    from zoneinfo import ZoneInfo
    vn = ZoneInfo("Asia/Ho_Chi_Minh")
    rs = make(tmp_path)
    missed = FORWARD_DAYS[:6]                                        # a week nobody ran the pipeline
    late = datetime(missed[-1].year, missed[-1].month, missed[-1].day, 20, 0, tzinfo=vn)
    with ResultsStore(tmp_path / "results.duckdb", rs) as store:
        result = run_daily(store, late, rule=RULE, n_random=2)
        paths = write_reports(store, result, late, tmp_path / "reports")
        first = paths[0].read_text()
        on_time = render_daily(store, result, late, day=missed[0])
    assert result.scanned == missed and [p.stem for p in paths] == [str(d) for d in missed]
    assert "Catch-up report" in first and "## Avoid list" in first and "## Today's events" in first
    assert "## Forward scoreboard" not in first and "| symbol | entry |" not in first
    latest = paths[-1].read_text()
    assert "Catch-up report" not in latest and "## Forward scoreboard" in latest
    assert first == on_time


def test_paper_portfolio_never_uses_an_incomplete_session(tmp_path):
    """The scan rejects a partial session; the paper portfolio must stop at the previous complete one."""
    from quant_research.daily_report import render_daily
    rs = make(tmp_path, drop_last_for="STK")
    with ResultsStore(tmp_path / "results.duckdb", rs) as store:
        r = run_daily(store, NOW, rule=RULE, n_random=2)
        dates = [d for (d,) in store.con.execute("SELECT DISTINCT run_date FROM paper_daily").fetchall()]
        last_day = store.con.execute("SELECT max(date) FROM paper_daily").fetchone()[0]
        report = render_daily(store, r, NOW)
    assert r.paper_date == FORWARD_DAYS[-2] and dates == [FORWARD_DAYS[-2]] and last_day == FORWARD_DAYS[-2]
    assert "latest complete session" in report
    rs = make(tmp_path)                                               # complete now: the paper portfolio catches up
    with ResultsStore(tmp_path / "results.duckdb", rs) as store:
        r = run_daily(store, NOW, rule=RULE, n_random=2)
        last_day = store.con.execute("SELECT max(date) FROM paper_daily WHERE run_date = ?", [r.paper_date]).fetchone()[0]
    assert r.paper_date == FORWARD_DAYS[-1] and last_day == FORWARD_DAYS[-1]


def test_paper_portfolio_stops_before_an_incomplete_session_in_the_middle(tmp_path):
    """complete -> incomplete -> complete: the re-simulation must not cross the incomplete session."""
    wh, rs = tmp_path / "w.duckdb", tmp_path / "r.duckdb"
    build_synthetic_warehouse(wh, start=START)
    middle = FORWARD_DAYS[-3]
    with Warehouse(wh) as w:
        w.connection.execute("DELETE FROM quotes_daily WHERE date = ? AND symbol = 'STK'", [middle])
    build(wh, rs, NOW)
    with ResultsStore(tmp_path / "results.duckdb", rs) as store:
        r = run_daily(store, NOW, rule=RULE, n_random=2)
        statuses = dict(store.con.execute("SELECT scan_date, status FROM daily_scans WHERE is_forward").fetchall())
        last_day = store.con.execute("SELECT max(date) FROM paper_daily WHERE run_date = ?", [r.paper_date]).fetchone()[0]
    assert statuses[middle] == "incomplete" and statuses[FORWARD_DAYS[-1]] == "scanned"
    assert r.paper_date == FORWARD_DAYS[-4] and last_day == FORWARD_DAYS[-4]


def test_paper_date_never_falls_before_the_forward_start_and_readers_follow_paper_state(tmp_path):
    """A pre-forward session is scanned, and the first forward session becomes incomplete: there is no valid
    paper record. paper_date is None, and readers do not show the older re-simulation as current."""
    import duckdb as _duckdb
    from stock_ui.exposure_data import paper_weights
    from stock_ui.research_data import _paper
    rs = make(tmp_path)
    with ResultsStore(tmp_path / "results.duckdb", rs) as store:
        first = run_daily(store, NOW, rule=RULE, n_random=2)                    # a valid record exists
    assert first.paper_date == FORWARD_DAYS[-1]
    pre = [d for d in calendar(START) if d < FORWARD_START][-1]
    wh, rs2 = tmp_path / "w.duckdb", tmp_path / "r.duckdb"
    with Warehouse(wh) as w:                                                    # the first forward day breaks
        w.connection.execute("DELETE FROM quotes_daily WHERE date = ? AND symbol = 'STK'", [FORWARD_DAYS[0]])
    build(wh, rs2, NOW)
    with ResultsStore(tmp_path / "results.duckdb", rs2) as store:
        run_daily(store, NOW, rescan=pre, rule=RULE, n_random=2)                # a scanned pre-forward session
        r = run_daily(store, NOW, rescan=FORWARD_DAYS[0], rule=RULE, n_random=2)
        state = store.con.execute("SELECT paper_date FROM paper_state").fetchone()[0]
        kept = store.con.execute("SELECT count(*) FROM paper_daily WHERE run_date = ?", [FORWARD_DAYS[-1]]).fetchone()[0]
    assert r.paper_date is None and state is None and kept > 0                  # history kept, not current
    con = _duckdb.connect(str(tmp_path / "results.duckdb"), read_only=True)
    try:
        assert _paper(con) == {"run_date": None}
    finally:
        con.close()

    class _Reader:                                                              # minimal Reader for paper_weights
        def read(self, _db, _key, fn):
            c = _duckdb.connect(str(tmp_path / "results.duckdb"), read_only=True)
            try:
                from stock_ui.db import ReadResult
                return ReadResult(fn(c), None, False, None)
            finally:
                c.close()
    assert paper_weights(_Reader()).value == {}


def test_one_forward_session_gives_no_current_paper_record(tmp_path):
    """With a single complete forward session there is nothing to simulate: paper_state must be NULL and no
    paper rows may remain for that run_date."""
    rs = make(tmp_path)
    with ResultsStore(tmp_path / "results.duckdb", rs) as store:
        run_daily(store, NOW, rule=RULE, n_random=2)
        store.con.execute("INSERT INTO paper_daily (run_date, date, equity) VALUES (?, ?, 1.0)",
                          [FORWARD_DAYS[0], FORWARD_DAYS[0]])                # a stale row for that run_date
        sessions = run_paper(store, FORWARD_DAYS[0], rule=RULE, n_random=2)
        left = store.con.execute("SELECT count(*) FROM paper_daily WHERE run_date = ?", [FORWARD_DAYS[0]]).fetchone()[0]
    assert sessions == 1 and left == 0
    wh, rs2 = tmp_path / "w.duckdb", tmp_path / "r.duckdb"
    with Warehouse(wh) as w:                                                  # only the first forward day complete
        w.connection.execute("DELETE FROM quotes_daily WHERE date = ? AND symbol = 'STK'", [FORWARD_DAYS[1]])
    build(wh, rs2, NOW)
    with ResultsStore(tmp_path / "results2.duckdb", rs2) as store:
        r = run_daily(store, NOW, rule=RULE, n_random=2)
        state = store.con.execute("SELECT paper_date FROM paper_state").fetchone()[0]
    assert r.paper_date == FORWARD_DAYS[0] and r.paper_sessions == 1 and state is None

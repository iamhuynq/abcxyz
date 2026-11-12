"""Page smoke tests with Streamlit's AppTest on a synthetic data directory.

Each page must render without exceptions, leave every database file untouched (read-only), and
never show the token.
"""

import base64
import json
import os
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from quant_research.backtest.data import UniverseRule
from quant_research.build import build
from quant_research.cross.build import build_cross
from quant_research.cross.params import CrossParams
from quant_research.daily import run_daily
from quant_research.results import ResearchParams, ResultsStore
from tests.research.synthetic import build_synthetic_warehouse

PAGES = Path(__file__).resolve().parents[2] / "src" / "stock_ui" / "pages"
NOW = datetime(2027, 3, 1, 19, 0, tzinfo=UTC)   # after the synthetic calendar: every session is final
RULE = UniverseRule(min_adv_value=0, min_session_index=0, min_stocks_per_date=2, signal_decile=3)


def _fake_jwt() -> str:
    def part(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")
    return f"{part({'alg': 'none'})}.{part({'exp': 1893456000, 'scope': ['orders-write', 'read']})}.SIGNATUREXYZ"


TOKEN = _fake_jwt()


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("ui") / "data"
    d.mkdir()
    build_synthetic_warehouse(d / "warehouse.duckdb", start=date(2026, 1, 5))
    build(d / "warehouse.duckdb", d / "research.duckdb", NOW)
    build_cross(d / "research.duckdb", d / "cross.duckdb", NOW,
                CrossParams(universe_size=30, min_adv_value=0, windows=(60, 120), cluster_window=120,
                            coint_window=120, n_large=2, start_year=2026))
    with ResultsStore(d / "results.duckdb", d / "research.duckdb") as store:
        run_daily(store, NOW, rule=RULE, n_random=2)
        store.start_run("run-a", "pattern", "research", ResearchParams(), NOW, ("p", "v1"))
        store.log_hypothesis("run-a", "lift_exec_excess_10d", "research", 0.01, NOW)
        store.finish_run("run-a")
    (d / "reports" / "daily").mkdir(parents=True)
    (d / "reports" / "daily" / "2026-10-05.md").write_text("# Daily report\n")
    return d


@pytest.fixture
def env(data_dir, tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(f"FIREANT_TOKEN={TOKEN}\n")
    monkeypatch.setenv("ENV_FILE", str(env_file))
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.delenv("FIREANT_TOKEN", raising=False)
    return data_dir


def _mtimes(data_dir: Path) -> dict[str, int]:
    return {p.name: p.stat().st_mtime_ns for p in data_dir.glob("*.duckdb*")}


def _rendered_text(at: AppTest) -> str:
    parts = []
    for kind in ("markdown", "caption", "code", "text", "title", "subheader", "metric", "success", "info",
                 "warning", "error"):
        for el in getattr(at, kind, []):
            parts.append(str(getattr(el, "value", "")))
            parts.append(str(getattr(el, "label", "")))
    for df in at.dataframe:
        parts.append(df.value.to_string())
    return "\n".join(parts)


@pytest.mark.parametrize("page", ["overview.py", "tasks.py", "symbol.py", "research.py", "backtest.py", "daily.py", "cross.py",
                                  "watch.py", "risk.py"])
def test_page_renders_read_only_without_token(env, page):
    before = _mtimes(env)
    at = AppTest.from_file(str(PAGES / page), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert _mtimes(env) == before
    text = _rendered_text(at)
    assert TOKEN not in text and "SIGNATUREXYZ" not in text


def test_overview_shows_facts(env):
    at = AppTest.from_file(str(PAGES / "overview.py"), default_timeout=60).run()
    labels = {m.label: m.value for m in at.metric}
    assert labels["Latest session"] != "-"
    assert labels["Token expires"] == "2030-01-01"
    assert any("orders-write" in c.value for c in at.caption)


def test_overview_and_watch_show_market_regimes(env):
    at = AppTest.from_file(str(PAGES / "overview.py"), default_timeout=60).run()
    assert not at.exception
    assert "Market regimes" in [h.value for h in at.subheader]
    assert len(at.get("plotly_chart")) == 1                                       # VNINDEX with risk shading
    assert any("direction:" in c.value and "risk:" in c.value for c in at.caption)
    w = AppTest.from_file(str(PAGES / "watch.py"), default_timeout=60).run()
    assert not w.exception and any(c.value.startswith("Regimes today") for c in w.caption)


def test_research_page_interactions_tab_without_a_scan(env):
    at = AppTest.from_file(str(PAGES / "research.py"), default_timeout=60).run()
    assert not at.exception and any("No interaction scan yet" in i.value for i in at.info)


def test_symbol_page_draws_the_chart(env):
    at = AppTest.from_file(str(PAGES / "symbol.py"), default_timeout=60).run()
    at.radio[0].set_value("All").run()
    assert not at.exception
    assert len(at.get("plotly_chart")) == 1
    at.toggle[0].set_value(False).run()                  # raw prices from the warehouse
    assert not at.exception and len(at.get("plotly_chart")) == 1


def test_overview_while_pipeline_runs_shows_info_not_error(env):
    from stock_ui import context
    context._reader.clear()                              # no cached reads: nothing earlier to show
    lock = env / ".daily.lock"
    lock.mkdir()
    (lock / "pid").write_text(str(os.getpid()))          # a live process holds it
    try:
        at = AppTest.from_file(str(PAGES / "overview.py"), default_timeout=60).run()
    finally:
        shutil.rmtree(lock)
    assert not at.exception
    assert any("pipeline is running" in m.value for m in [*at.info, *at.warning])


def test_research_page_run_detail(env):
    at = AppTest.from_file(str(PAGES / "research.py"), default_timeout=60).run()
    at.selectbox[0].set_value("run-a").run()
    assert not at.exception
    assert any("run-a" in str(df.value.to_string()) for df in at.dataframe)


def test_research_page_registry_tab(env):
    at = AppTest.from_file(str(PAGES / "research.py"), default_timeout=60).run()
    assert not at.exception
    assert {m.label for m in at.metric} >= {"failed", "rejected", "forward", "monitoring", "candidate"}
    reg = next(df.value for df in at.dataframe if "hypothesis_id" in df.value.columns)
    assert len(reg) == 15 and "Validation decisions" not in [t.label for t in at.tabs]
    pick = next(s for s in at.selectbox if s.label == "Hypothesis")
    pick.set_value("P3-C3").run()
    assert not at.exception
    history = next(df.value for df in at.dataframe if "seq" in df.value.columns)
    assert list(history["status"]) == ["research", "candidate", "preregistered", "validated", "forward"]


def test_backtest_page_shows_the_paper_portfolio(env):
    at = AppTest.from_file(str(PAGES / "backtest.py"), default_timeout=60).run()
    assert not at.exception
    assert {m.label for m in at.metric} >= {"Strategy", "Random median", "Equal weight", "VNINDEX"}
    assert len(at.get("plotly_chart")) == 1
    assert {m.label for m in at.metric} >= {"Market beta", "Effective bets"}     # exposure of the open positions
    assert any("left out): DEL" in c.value for c in at.caption)                   # delisted: no returns, reported


def test_daily_page_lists_events(env):
    at = AppTest.from_file(str(PAGES / "daily.py"), default_timeout=60).run()
    assert not at.exception
    assert any("Daily report" in m.value for m in at.markdown)
    assert len(at.dataframe) == 2                              # events of the latest session, scan status


def test_invalidate_through_the_task_runner_end_to_end(env, monkeypatch):
    """The real `quant runs --invalidate` CLI, started like the Research page does, under the shared lock."""
    import sys
    import time

    import duckdb

    from stock_ui import tasks
    monkeypatch.setenv("QUANT_CMD", f"{sys.executable} -m quant_research.cli")
    ui_run = tasks.start(env, "invalidate_run", run_id="run-a", reason="test: definition bug")
    deadline = time.time() + 60
    while tasks.load_record(env, ui_run).status() not in ("ok", "failed", "refused") and time.time() < deadline:
        time.sleep(0.2)
    rec = tasks.load_record(env, ui_run)
    assert rec.status() == "ok", tasks.tail(env, ui_run)
    con = duckdb.connect(str(env / "results.duckdb"), read_only=True)
    try:
        row = con.execute("SELECT status, invalid_reason FROM research_runs WHERE run_id = 'run-a'").fetchone()
    finally:
        con.close()
    assert row == ("invalid", "test: definition bug")
    missing = tasks.start(env, "invalidate_run", run_id="no-such-run", reason="test: unknown run")
    deadline = time.time() + 60
    while tasks.load_record(env, missing).status() not in ("ok", "failed", "refused") and time.time() < deadline:
        time.sleep(0.2)
    assert tasks.load_record(env, missing).exit_code == 1               # unknown run: reported, not "ok"


def test_cross_page_peers_and_pair_chart(env):
    at = AppTest.from_file(str(PAGES / "cross.py"), default_timeout=60).run()
    assert not at.exception                                     # default: research period only (synthetic: empty)
    assert not any("2024 onward" in w.value for w in at.warning)
    at.toggle[0].set_value(True).run()                          # include later data
    assert not at.exception and any("2024 onward" in w.value for w in at.warning)
    assert len(at.get("plotly_chart")) == 1                     # pair tab: rolling correlation
    assert at.dataframe[0].value.shape[0] >= 1                  # peers of the default symbol


def test_watch_page_watchlist_and_staleness(env, monkeypatch):
    import fireant_crawler.sessions as sessions
    from stock_ui import watchlist
    monkeypatch.setattr(sessions, "final_session", lambda now: date(2099, 1, 1))     # data is always behind
    watchlist.save(env, ["STK", "UPC"], datetime(2026, 10, 8, tzinfo=UTC))
    try:
        at = AppTest.from_file(str(PAGES / "watch.py"), default_timeout=60).run()
        assert not at.exception, [e.value for e in at.exception]
        assert any("latest final session is 2099-01-01" in w.value for w in at.warning)
        bets = {m.label: m.value for m in at.metric}["Effective bets"]              # exposure panel, equal weights
        assert bets.endswith("of 2") and 1.0 <= float(bets.split()[0]) <= 2.0
        assert any("sessions to 2027-02-26." in c.value for c in at.caption)
        assert "STK" in at.text_area[0].value and "UPC" in at.text_area[0].value
        at.text_area[0].set_value("upc, nope!, STK STK")                  # form: value and submit in one run
        at.button[0].click().run()
        assert watchlist.load(env) == ["UPC", "STK"]
        assert any("NOPE!" in w.value for w in at.warning)            # symbols are upper-cased
    finally:
        watchlist.path(env).unlink(missing_ok=True)


def test_risk_page_without_runs(env):
    at = AppTest.from_file(str(PAGES / "risk.py"), default_timeout=60).run()
    assert not at.exception and any("No risk evaluation yet" in i.value for i in at.info)


@pytest.fixture
def risk_env(data_dir, tmp_path, monkeypatch):
    """A copy of the UI data with one complete paper risk run and one failed risk run."""
    from quant_research.risk import data, evaluate, limits
    d = tmp_path / "data"
    shutil.copytree(data_dir, d)
    with ResultsStore(d / "results.duckdb", d / "research.duckdb") as store:
        store.con.execute(f"ATTACH '{(d / 'cross.duckdb').as_posix()}' AS cx (READ_ONLY)")
        evaluate.evaluate(store, data.paper_snapshot(store.con), limits.DEFAULT, NOW)
        store.start_run("broken-risk", "risk", "forward", ResearchParams(), NOW)
        store.con.execute("UPDATE research_runs SET status = 'failed' WHERE run_id = 'broken-risk'")
    env_file = tmp_path / ".env"
    env_file.write_text(f"FIREANT_TOKEN={TOKEN}\n")
    monkeypatch.setenv("ENV_FILE", str(env_file))
    monkeypatch.setenv("DATA_DIR", str(d))
    return d


def test_risk_page_shows_a_stored_run_read_only(risk_env):
    before = _mtimes(risk_env)
    at = AppTest.from_file(str(PAGES / "risk.py"), default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]
    assert _mtimes(risk_env) == before
    text = _rendered_text(at)
    assert "policy `default/v2`" in text.lower() or "Policy `default/v2`" in text
    assert "scenario set `stress/v1`" in text and "risk model `risk-v2`" in text
    assert "broken-risk (failed)" in text                                    # a failed run is listed, not hidden
    tables = [df.value for df in at.dataframe]
    stress = next(t for t in tables if "scenario_id" in t.columns)
    assert set(stress["status"]) <= {"MEASURED", "INCOMPLETE"} and {"market", "suspension"} <= set(stress["family"])
    limits_table = next(t for t in tables if "limit_name" in t.columns)
    assert {"max_drawdown_historical", "current_drawdown_write_down"} <= set(limits_table["metric"])
    incomplete = limits_table[limits_table["status"] == "INCOMPLETE"]
    assert len(incomplete) and incomplete["reason"].str.len().gt(0).all()   # INCOMPLETE is shown with its reason
    assert any("liquidation_days" in t.columns for t in tables)

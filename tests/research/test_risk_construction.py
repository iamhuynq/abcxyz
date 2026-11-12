"""Stress persistence and the construction-grid re-simulation on a synthetic store
(docs/risk-engine-pr2-plan.md, D1, D2, D8)."""

import json
import time
from dataclasses import asdict, replace
from datetime import UTC, date, datetime, timedelta

import duckdb
import pytest

from quant_research import cli, daily
from quant_research.backtest import runner
from quant_research.backtest.data import UniverseRule
from quant_research.build import build
from quant_research.factors import FactorParams
from quant_research.portfolio import evaluate as grid
from quant_research.portfolio.construction import Construction
from quant_research.regimes import RegimeParams
from quant_research.results import ResearchParams, ResultsStore
from quant_research.risk import construction, data, evaluate, limits, report, stress
from tests.research.synthetic import build_synthetic_warehouse, calendar

NOW = datetime(2026, 11, 12, 9, 0, tzinfo=UTC)
START = date(2022, 1, 3)
DAYS = calendar(START)
CFG = grid.Config("liquidity", "weekly", Construction(n_names=2, entry_rank=2, exit_rank=2, max_position_weight=0.6,
                                                       max_industry_weight=1.0, min_adv_value=0.0))
RULE = UniverseRule(min_adv_value=0.0, min_session_index=0, min_stocks_per_date=1)
RUN = "20261009T171446-portfolio-T1-research"


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
        store.con.execute(grid.SCHEMA)
        _grid_run(store)
    return d


def _grid_run(store: ResultsStore, cfg: grid.Config = CFG, version: str | None = None, shift: float = 0.0,
              name: str = "T1", run_id: str = RUN) -> None:
    """A stored grid run of CFG whose decision-setting metrics are this code's own (shifted by `shift`)."""
    store.start_run(run_id, "portfolio", "research", ResearchParams(extra={"config": asdict(cfg),
                    "universe": asdict(RULE), "capitals": [1e9], "n_control": 0}), NOW,
                    (name, version or grid.config_version(cfg)))
    prep = grid.prepare(store, cfg, "research", RULE)
    res = grid.simulate(prep, cfg)
    m = grid._metrics(res, 1e9)
    store.con.executemany("INSERT INTO portfolio_results VALUES (?, ?, 1e9, 'v1_k1', 'strategy', ?, ?)",
                          [(run_id, name, k, float(v) * (1 + shift) if k == "cagr" else float(v)) for k, v in m.items()])
    store.finish_run(run_id)


def _tables(path) -> dict:
    with duckdb.connect(str(path), read_only=True) as con:
        return {t: con.execute(f"SELECT * FROM {t} ORDER BY ALL").fetchall()
                for t in ("portfolio_results", "hypothesis_log", "hypotheses", "hypothesis_transitions")} | {
            "grid_runs": con.execute("SELECT * FROM research_runs WHERE kind = 'portfolio' ORDER BY ALL").fetchall()}


def test_resimulation_is_deterministic_reconciles_and_writes_nothing_to_the_grid(env):
    before = _tables(env / "results.duckdb")
    with ResultsStore(env / "results.duckdb", env / "research.duckdb") as store:
        snap, src, rec = construction.snapshot(store, "T1")
        again, _, rec2 = construction.snapshot(store, "T1")
        assert snap.positions.equals(again.positions) and rec == rec2 and snap.nav == again.nav
        assert snap.target == "construction:T1@resim" and src.run_id == RUN
        assert set(snap.curves) == {"default marks (last traded close)", "write-down 60"}
        assert snap.reconciliation_gap == 0.0 and len(snap.positions) == 2 and all(w for *_r, w in rec)
        assert "reconciles with it" in snap.notes[0] and "NOT the historical run" in snap.notes[0]
        mid, _, _ = construction.snapshot(store, "T1", DAYS[200])
        assert mid.valuation_date == DAYS[200]
        run_id, a = evaluate.evaluate(store, snap, limits.DEFAULT, NOW, extra={"source_run": src.run_id},
                                      reconciliation=(src.run_id, rec))
        stored = store.con.execute("SELECT count(*), bool_and(within) FROM risk_reconciliation WHERE run_id = ?",
                                   [run_id]).fetchone()
        text = report.render(store, run_id)
        params = json.loads(store.con.execute("SELECT params FROM research_runs WHERE run_id = ?",
                                              [run_id]).fetchone()[0])
    assert stored == (len(rec), True) and "It reconciles with the stored run" in text
    assert params["extra"]["source_run"] == RUN and params["extra"]["scenarios"] == "stress/v1"
    assert _tables(env / "results.duckdb") == before


def test_a_resimulation_that_differs_is_labelled_not_hidden(env):
    with ResultsStore(env / "results.duckdb", env / "research.duckdb") as store:
        store.con.execute("UPDATE portfolio_results SET value = value * 1.001 WHERE metric = 'cagr'")
        snap, src, rec = construction.snapshot(store, "T1")
        bad = [m for m, *_r, w in rec if not w]
        run_id, _a = evaluate.evaluate(store, snap, limits.DEFAULT, NOW, reconciliation=(src.run_id, rec))
        text = report.render(store, run_id)
    assert bad == ["cagr"] and "does not reconcile with it: cagr differ" in snap.notes[0]
    assert "does NOT reconcile (1 of" in text and "| cagr |" in text


def test_a_changed_or_missing_configuration_is_refused(env):
    with ResultsStore(env / "results.duckdb", env / "research.duckdb") as store:
        with pytest.raises(ValueError, match="no valid construction run"):
            construction.snapshot(store, "C9")
        store.con.execute("UPDATE research_runs SET pattern_version = 'deadbeef' WHERE run_id = ?", [RUN])
        with pytest.raises(ValueError, match="refusing to re-simulate"):
            construction.snapshot(store, "T1")


def test_stress_results_are_stored_with_the_frozen_scenario_set(env):
    with ResultsStore(env / "results.duckdb", env / "research.duckdb") as store:
        snap, src, rec = construction.snapshot(store, "T1")
        run_id, a = evaluate.evaluate(store, snap, limits.DEFAULT, NOW)
        rows = store.con.execute("SELECT seq, scenario_id, family, status FROM risk_stress_results WHERE run_id = ? "
                                 "ORDER BY seq", [run_id]).fetchall()
        assert [r[0] for r in rows] == list(range(len(rows)))
        assert {r[2] for r in rows} == {"market", "industry", "cluster", "liquidity", "suspension"}
        assert [r[1] for r in rows][:2] == ["market_beta_-5%", "market_uniform_-5%"]          # declared order
        changed = replace(stress.STRESS_V1, adv_factor=0.25)
        with pytest.raises(ValueError, match="new version"):
            evaluate.evaluate(store, snap, limits.DEFAULT, NOW + timedelta(minutes=1), scenarios=changed)
        none_id, _ = evaluate.evaluate(store, snap, limits.DEFAULT, NOW + timedelta(minutes=2), scenarios=None)
        assert store.con.execute("SELECT count(*) FROM risk_stress_results WHERE run_id = ?",
                                 [none_id]).fetchone()[0] == 0
        assert "Not run for this evaluation" in report.render(store, none_id)
        assert store.con.execute("SELECT count(*) FROM hypothesis_log WHERE run_id LIKE '%-risk-%'").fetchone()[0] == 0


def test_cli_construction_and_no_stress(env, capsys):
    assert cli.main(["risk", "evaluate", "--construction", "T1"]) == 0
    out = capsys.readouterr().out
    assert "Re-simulation of T1" in out and "construction:T1@resim" in out
    assert cli.main(["risk", "evaluate", "--construction", "C9"]) == 1
    time.sleep(1.05)                                     # run ids have one-second resolution
    assert cli.main(["risk", "evaluate", "--construction", "T1", "--no-stress", "--date",
                     DAYS[150].isoformat()]) == 0
    reports = sorted((env / "reports" / "research").glob("risk-*.md"))
    assert len(reports) == 2 and "Not run for this evaluation" in reports[-1].read_text()
    capsys.readouterr()


def test_the_stored_configuration_is_used_not_the_code_constant(env):
    """A run named C1 that stored another configuration than CONFIGS['C1'] is re-simulated as stored."""
    assert grid.CONFIGS["C1"] != CFG
    with ResultsStore(env / "results.duckdb", env / "research.duckdb") as store:
        _grid_run(store, name="C1", run_id="20261009T171446-portfolio-C1-research")
        snap, src, rec = construction.snapshot(store, "C1")
    assert src.config == CFG and len(snap.positions) == 2 and all(w for *_r, w in rec)

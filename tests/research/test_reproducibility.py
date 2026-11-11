"""Reproducibility baseline (docs/research-integrity.md, workstream H).

The fixture experiment: build research.duckdb twice from the same warehouse, and run the same pattern, base
comparison and decile scan twice on the same research database. Every stored output must be equal.

- Exact: every key, count, integer, boolean, label and string (a regime state, a decile, an event type, the row
  set itself). These decide what a statistic is computed on.
- Within a relative 1e-9: floating-point values. DuckDB aggregates in parallel, so the order of a sum (window
  sums in market_regimes, per-date corr() in the decile scan) can change between runs. Measured on this fixture:
  at most 3.4e-16 in the build and 2.6e-13 in scan IC statistics; everything else was bit-identical.
- Excluded on purpose: run ids, created / logged timestamps and build timestamps (they name the run, not
  its result), and the build duration.
- Known exception: a decile pattern whose feature has ties at the decile boundary (`patterns.py` ntile
  without a tie-break); see finding H-2 in docs/research-integrity.md.

Run it: `uv run pytest tests/research/test_reproducibility.py -q`.
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd

from quant_research.build import build
from quant_research.engine import run_pattern, run_scan
from quant_research.patterns import Pattern
from quant_research.provenance import collect, environment
from quant_research.results import ResearchParams, ResultsStore, research_code_hash
from tests.research.synthetic import build_synthetic_warehouse
from tests.research.synthetic_research import build_research_db

NOW = datetime(2026, 11, 11, 9, 0, tzinfo=UTC)
PARAMS = ResearchParams(min_adv_value=1e9, min_session_index=20, min_stocks_per_date=30)
BASE = Pattern("signal_high", "volume_ratio_20 > 0.8", "planted signal")
REFINED = Pattern("signal_high_noise", "volume_ratio_20 > 0.8 AND return_1d > 0.5", "refinement", base="signal_high")
RTOL, ATOL = 1e-9, 1e-15
RUN_TABLES = ("pattern_stats", "pattern_lifts", "pattern_comparisons", "pattern_occurrences", "scan_stats",
              "scan_deciles", "hypothesis_log")
NOT_RESULTS = ("run_id", "logged_at", "built_at", "build_id", "seconds")


def _frame(path: Path, table: str) -> pd.DataFrame:
    with duckdb.connect(str(path), read_only=True) as con:
        df = con.execute(f"SELECT * FROM {table}").df()
    df = df.drop(columns=[c for c in NOT_RESULTS if c in df.columns])
    return df.sort_values(list(df.columns)).reset_index(drop=True)


def test_build_twice_gives_identical_tables(tmp_path):
    wh = tmp_path / "w.duckdb"
    build_synthetic_warehouse(wh)
    a = tmp_path / "a" / "r.duckdb"
    b = tmp_path / "b" / "r.duckdb"
    build(wh, a, NOW)
    build(wh, b, NOW + timedelta(hours=1))
    with duckdb.connect(str(a), read_only=True) as con:
        tables = [t for (t,) in con.execute("SELECT table_name FROM information_schema.tables "
                                            "WHERE table_type = 'BASE TABLE'").fetchall()]
    assert len(tables) >= 10
    for table in tables:
        pd.testing.assert_frame_equal(_frame(a, table), _frame(b, table), rtol=RTOL, atol=ATOL, obj=table)


def _experiment(results: Path, research: Path, now: datetime) -> dict:
    with ResultsStore(results, research) as store:
        lib = {p.name: p for p in (BASE, REFINED)}
        run_pattern(store, BASE, "research", PARAMS, now, library=lib)
        run_pattern(store, REFINED, "research", PARAMS, now, library=lib)
        run_scan(store, "research", PARAMS, now, features=("volume_ratio_20", "return_1d"))
        q = store.con.execute("SELECT label, p_value, q_value, q_by FROM hypothesis_q ORDER BY label").df()
        params = [p for (p,) in store.con.execute("SELECT params FROM research_runs ORDER BY run_id").fetchall()]
    return {"q": q, "params": params}


def test_same_experiment_twice_gives_identical_results(tmp_path):
    research = tmp_path / "research.duckdb"
    build_research_db(research)
    first = _experiment(tmp_path / "one.duckdb", research, NOW)
    second = _experiment(tmp_path / "two.duckdb", research, NOW + timedelta(days=1))
    assert first["params"] == second["params"] and len(first["q"]) > 10
    pd.testing.assert_frame_equal(first["q"], second["q"], rtol=RTOL, atol=ATOL, obj="hypothesis_q")
    for table in RUN_TABLES:
        pd.testing.assert_frame_equal(_frame(tmp_path / "one.duckdb", table), _frame(tmp_path / "two.duckdb", table),
                                      rtol=RTOL, atol=ATOL, obj=table)


def test_provenance_records_the_environment(tmp_path):
    research = tmp_path / "research.duckdb"
    build_research_db(research)
    with ResultsStore(tmp_path / "results.duckdb", research) as store:
        p = collect(store.con, research_code_hash())
    assert p["env"] == environment()
    assert environment().startswith("python 3.") and "duckdb " in environment() and "numpy " in environment()

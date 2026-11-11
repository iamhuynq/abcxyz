"""Pattern engine, feature scan, hypothesis log and holdout gate."""

from datetime import UTC, datetime

import duckdb
import pytest

from quant_research.build import build, file_hash
from quant_research.engine import run_pattern, run_scan
from quant_research.patterns import LIBRARY, Pattern
from quant_research.report import render_pattern_batch, render_scan
from quant_research.results import HoldoutLockedError, ResearchParams, ResultsStore
from quant_research.stats import benjamini_hochberg, benjamini_yekutieli
from tests.research.synthetic import build_synthetic_warehouse
from tests.research.synthetic_research import build_research_db

NOW = datetime(2026, 10, 4, 20, 0, tzinfo=UTC)
PARAMS = ResearchParams(min_adv_value=1e9, min_session_index=20, min_stocks_per_date=30)
SIGNAL = Pattern("signal_high", "volume_ratio_20 > 0.8", "planted signal")
NOISE = Pattern("noise_high", "return_1d > 0.8", "pure noise")
REFINED = Pattern("signal_high_noise", "volume_ratio_20 > 0.8 AND return_1d > 0.5", "noise refinement", base="signal_high")
LIB = {p.name: p for p in (SIGNAL, NOISE, REFINED)}


@pytest.fixture
def store(tmp_path):
    research = tmp_path / "research.duckdb"
    build_research_db(research)
    s = ResultsStore(tmp_path / "results.duckdb", research)
    s.research_path = research  # type: ignore[attr-defined]
    yield s
    s.close()


def stat(store, run_id, horizon="exec_excess_10d", segment="all"):
    cur = store.con.execute("SELECT * FROM pattern_stats WHERE run_id = ? AND horizon = ? AND segment = ?",
                            [run_id, horizon, segment])
    return dict(zip([d[0] for d in cur.description], cur.fetchone()))


def test_pattern_detects_planted_signal_and_matches_direct_sql(store):
    run_id = run_pattern(store, SIGNAL, "research", PARAMS, NOW, library=LIB)
    s = stat(store, run_id)
    direct = store.con.execute("""SELECT count(*), count(DISTINCT date), avg(m) FROM (
        SELECT date, fwd_excess_exec_10d, avg(fwd_excess_exec_10d) OVER (PARTITION BY date) AS m
        FROM rs.feature_target WHERE period = 'research' AND volume_ratio_20 > 0.8)""").fetchone()
    date_mean = store.con.execute("""SELECT avg(m) FROM (SELECT date, avg(fwd_excess_exec_10d) m
        FROM rs.feature_target WHERE period = 'research' AND volume_ratio_20 > 0.8 GROUP BY date)""").fetchone()[0]
    assert (s["n_events"], s["n_dates"]) == direct[:2]
    assert s["mean"] == pytest.approx(date_mean)
    assert s["mean"] == pytest.approx(0.0225, abs=0.003)          # 0.05 * (0.9 - 0.5)
    assert s["p"] < 1e-10 and s["mean_after_cost"] == pytest.approx(s["mean"] - 0.004)
    assert store.con.execute("SELECT count(*) FROM pattern_occurrences WHERE run_id = ?", [run_id]).fetchone()[0] == s["n_events"]


def test_lift_is_pattern_minus_universe_on_the_same_dates(store):
    run_id = run_pattern(store, SIGNAL, "research", PARAMS, NOW, library=LIB)
    lift, base, t = store.con.execute("""SELECT lift, baseline_mean, t FROM pattern_lifts
                                         WHERE run_id = ? AND horizon = 'exec_excess_10d'""", [run_id]).fetchone()
    expected = store.con.execute("""
        WITH u AS (SELECT date, avg(fwd_excess_exec_10d) um FROM rs.feature_target WHERE period = 'research' GROUP BY date),
             p AS (SELECT date, avg(fwd_excess_exec_10d) pm FROM rs.feature_target
                   WHERE period = 'research' AND volume_ratio_20 > 0.8 GROUP BY date)
        SELECT avg(pm - um), avg(um) FROM p JOIN u USING (date)""").fetchone()
    assert lift == pytest.approx(expected[0]) and base == pytest.approx(expected[1])
    assert lift == pytest.approx(0.02, abs=0.003) and t > 10
    assert set(store.q_values(run_id)) == {f"lift_exec_excess_{h}d" for h in (5, 10, 20)}


def test_noise_pattern_is_not_significant_after_bh(store):
    run_pattern(store, SIGNAL, "research", PARAMS, NOW, library=LIB)
    noise_id = run_pattern(store, NOISE, "research", PARAMS, NOW, library=LIB)
    q = store.q_values(noise_id)
    assert len(q) == 3 and all(v > 0.05 for v in q.values())


def test_refinement_without_information_is_detected_as_such(store):
    run_pattern(store, SIGNAL, "research", PARAMS, NOW, library=LIB)
    run_id = run_pattern(store, REFINED, "research", PARAMS, NOW, library=LIB)
    comps = store.con.execute("SELECT horizon, n_dates, p FROM pattern_comparisons WHERE run_id = ?", [run_id]).fetchall()
    assert len(comps) == 3 and all(n > 100 for _, n, _ in comps)
    assert all(store.q_values(run_id)[f"vs_base_{h}"] > 0.05 for h, _, _ in comps)


def test_holdout_is_locked_and_final_runs_are_counted(store):
    with pytest.raises(HoldoutLockedError):
        run_pattern(store, SIGNAL, "holdout", PARAMS, NOW, library=LIB)
    assert store.con.execute("SELECT count(*) FROM research_runs").fetchone()[0] == 0
    run_pattern(store, SIGNAL, "holdout", PARAMS, NOW, final=True, library=LIB)
    assert store.holdout_runs() == 1


def test_q_values_match_python_bh_and_exclude_invalid_runs(store):
    a = run_pattern(store, SIGNAL, "research", PARAMS, NOW, library=LIB)
    b = run_pattern(store, NOISE, "research", PARAMS, datetime(2026, 10, 4, 21, tzinfo=UTC), library=LIB)
    store.invalidate(b, "test")
    logged = store.con.execute("""SELECT h.label, h.p_value FROM hypothesis_log h JOIN research_runs r USING (run_id)
                                  WHERE r.status = 'ok' ORDER BY h.label""").fetchall()
    expected = dict(zip([l for l, _ in logged], benjamini_hochberg([p for _, p in logged])))
    assert store.q_values(a) == pytest.approx(expected)
    assert store.q_values(b) == {}

    by = dict(store.con.execute("SELECT label, q_by FROM hypothesis_q").fetchall())
    assert by == pytest.approx(dict(zip([l for l, _ in logged], benjamini_yekutieli([p for _, p in logged]))))
    assert all(by[label] >= q for label, q in expected.items())


def test_scan_ranks_the_signal_feature_and_not_the_noise(store):
    run_id = run_scan(store, "research", PARAMS, NOW, features=("volume_ratio_20", "return_1d"))
    rows = {(f, h): (sp, t, ic, mono) for f, h, sp, t, ic, mono in store.con.execute(
        "SELECT feature, horizon, spread, spread_t, ic_mean, monotonicity FROM scan_stats WHERE run_id = ? AND segment = 'all'",
        [run_id]).fetchall()}
    sp, t, ic, mono = rows[("volume_ratio_20", "exec_excess_10d")]
    assert sp == pytest.approx(0.045, abs=0.006) and t > 10 and ic > 0.3 and mono > 0.95
    q = store.q_values(run_id)
    assert all(q[f"spread_return_1d_exec_excess_{h}d"] > 0.05 for h in (5, 10, 20))
    assert "volume_ratio_20" in render_scan(store.con, run_id, NOW)


def test_scan_ic_is_exact_spearman_with_ties_and_deciles_are_deterministic(store):
    """A coarse feature (many ties): the scan IC must equal scipy's Spearman (average ranks) per date, and the
    decile means must be identical across runs (ties broken by symbol)."""
    import numpy as np
    import scipy.stats as sps
    table = "(SELECT *, round(volume_ratio_20 * 4, 0) AS coarse FROM rs.feature_target)"   # 5 unequal tie groups
    ids = [run_scan(store, "research", PARAMS, NOW.replace(minute=m), features=("coarse",), table=table, name=f"ties{m}")
           for m in (1, 2)]
    got = store.con.execute("""SELECT ic_mean FROM scan_stats WHERE run_id = ? AND feature = 'coarse'
                               AND horizon = 'exec_excess_10d' AND segment = 'all'""", [ids[0]]).fetchone()[0]
    rows = store.con.execute(f"""SELECT date, coarse, fwd_excess_exec_10d FROM {table}
        WHERE {PARAMS.universe_sql()} AND period = 'research' AND coarse IS NOT NULL AND isfinite(coarse)
          AND fwd_excess_exec_10d IS NOT NULL""").df()
    ics = []
    for _, g in rows.groupby("date"):
        if len(g) >= PARAMS.min_stocks_per_date:
            v = sps.spearmanr(g["coarse"], g["fwd_excess_exec_10d"]).statistic
            if np.isfinite(v):
                ics.append(v)
    assert rows.groupby("date")["coarse"].nunique().mean() < rows.groupby("date").size().mean() / 3   # many ties
    assert got == pytest.approx(np.mean(ics), rel=1e-9)
    deciles = [store.con.execute("""SELECT feature, horizon, segment, decile, mean FROM scan_deciles WHERE run_id = ?
                                    ORDER BY ALL""", [r]).fetchall() for r in ids]
    assert deciles[0] == deciles[1] and deciles[0]


def test_runs_do_not_modify_the_research_database(store):
    before = file_hash(store.research_path)
    run_pattern(store, SIGNAL, "research", PARAMS, NOW, library=LIB)
    run_scan(store, "research", PARAMS, NOW, features=("volume_ratio_20",))
    assert file_hash(store.research_path) == before


def test_end_to_end_on_phase2_build_with_library_patterns(tmp_path):
    """Real feature_target columns from the Phase 2 build must satisfy every library pattern and the report."""
    warehouse, research = tmp_path / "warehouse.duckdb", tmp_path / "research.duckdb"
    build_synthetic_warehouse(warehouse)
    build(warehouse, research, NOW)
    relaxed = ResearchParams(min_adv_value=0, min_session_index=0, min_stocks_per_date=2)
    with ResultsStore(tmp_path / "results.duckdb", research) as store:
        run_ids = [run_pattern(store, p, "research", relaxed, NOW) for p in LIBRARY.values()]
        report = render_pattern_batch(store.con, run_ids, "research", NOW)
        scan_id = run_scan(store, "research", relaxed, NOW)
        n_scan = store.con.execute("SELECT count(*) FROM scan_stats WHERE run_id = ?", [scan_id]).fetchone()[0]
    assert "drop3_volume2" in report and n_scan == 30 * 3 * 7


def test_null_outcomes_are_missing_not_zero(store):
    """Regression: masked NULLs from fetchnumpy must become NaN, never 0."""
    store.con.execute("DETACH rs")
    con = duckdb.connect(str(store.research_path))
    con.execute("UPDATE feature_target SET fwd_excess_exec_10d = NULL WHERE volume_ratio_20 > 0.8 AND symbol IN ('S1', 'S2', 'S3')")
    expected = con.execute("""SELECT count(fwd_excess_exec_10d), avg(m) FROM (SELECT fwd_excess_exec_10d,
        avg(fwd_excess_exec_10d) OVER (PARTITION BY date) m FROM feature_target
        WHERE period = 'research' AND volume_ratio_20 > 0.8)""").fetchone()
    date_mean = con.execute("""SELECT avg(m) FROM (SELECT date, avg(fwd_excess_exec_10d) m FROM feature_target
        WHERE period = 'research' AND volume_ratio_20 > 0.8 AND fwd_excess_exec_10d IS NOT NULL GROUP BY date)""").fetchone()[0]
    con.close()
    store.con.execute(f"ATTACH '{store.research_path.as_posix()}' AS rs (READ_ONLY)")
    run_id = run_pattern(store, SIGNAL, "research", PARAMS, NOW, library=LIB)
    s = stat(store, run_id)
    assert s["n_events"] == expected[0]
    assert s["mean"] == pytest.approx(date_mean)


def test_ic_ignores_zero_variance_days(store):
    """Regression: corr() returns NaN (not NULL) when a feature is constant on a date; one NaN day must not
    turn the whole IC into NaN."""
    store.con.execute("DETACH rs")
    con = duckdb.connect(str(store.research_path))
    con.execute("UPDATE feature_target SET volume_ratio_20 = 0.5 WHERE date = DATE '2020-01-10'")
    con.close()
    store.con.execute(f"ATTACH '{store.research_path.as_posix()}' AS rs (READ_ONLY)")
    run_id = run_scan(store, "research", PARAMS, NOW, features=("volume_ratio_20",))
    ic, ic_t = store.con.execute("""SELECT ic_mean, ic_t FROM scan_stats WHERE run_id = ? AND segment = 'all'
                                    AND horizon = 'exec_excess_10d'""", [run_id]).fetchone()
    assert ic == ic and ic > 0.3 and ic_t > 10      # ic == ic is False for NaN


def test_decile_pattern_selects_top_decile_per_date_and_compares_with_base(store):
    top = Pattern("signal_d10", "TRUE", "top decile of the planted signal", decile_of="volume_ratio_20", decile=10)
    refined = Pattern("signal_d10_noise", "return_1d > 0.5", "noise refinement of the top decile",
                      base="signal_d10", decile_of="volume_ratio_20", decile=10)
    lib = {"signal_d10": top, "signal_d10_noise": refined}
    run_id = run_pattern(store, top, "research", PARAMS, NOW, library=lib)
    n_events, n_dates = store.con.execute("""SELECT n_events, n_dates FROM pattern_stats
        WHERE run_id = ? AND horizon = 'exec_excess_10d' AND segment = 'all'""", [run_id]).fetchone()
    expected = store.con.execute("""SELECT count(*), count(DISTINCT date) FROM (
        SELECT date, ntile(10) OVER (PARTITION BY date ORDER BY volume_ratio_20) d FROM rs.feature_target
        WHERE period = 'research') WHERE d = 10""").fetchone()
    assert (n_events, n_dates) == expected
    lift = store.con.execute("SELECT lift FROM pattern_lifts WHERE run_id = ? AND horizon = 'exec_excess_10d'", [run_id]).fetchone()[0]
    assert lift == pytest.approx(0.05 * (0.95 - 0.5), abs=0.003)

    ref_id = run_pattern(store, refined, "research", PARAMS, NOW, library=lib)
    n_cmp, p_cmp = store.con.execute("""SELECT n_dates, p FROM pattern_comparisons
        WHERE run_id = ? AND horizon = 'exec_excess_10d'""", [ref_id]).fetchone()
    assert n_cmp > 100 and p_cmp > 0.01                     # noise refinement adds nothing
    definition = store.con.execute("SELECT where_sql FROM pattern_definitions WHERE name = 'signal_d10'").fetchone()[0]
    assert "decile 10 of volume_ratio_20" in definition

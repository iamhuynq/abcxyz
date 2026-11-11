"""The FDR family (`hypothesis_q`): who is in it, and edge cases of the p-values that enter it.

Family: every logged test with a p-value of every run whose status is 'ok'. A test whose p-value could not be
computed stays logged with NULL; running, failed and invalid runs are outside the family.
"""

import math
from datetime import UTC, datetime

import pytest

from quant_research.results import ResultsStore
from quant_research.stats import benjamini_hochberg, benjamini_yekutieli
from tests.research.synthetic_research import build_research_db

NOW = datetime(2026, 11, 11, 9, tzinfo=UTC)


@pytest.fixture
def store(tmp_path):
    research = tmp_path / "research.duckdb"
    build_research_db(research)
    s = ResultsStore(tmp_path / "results.duckdb", research)
    yield s
    s.close()


def _run(store, run_id, status="ok"):
    store.con.execute("INSERT INTO research_runs VALUES (?, 'scan', NULL, NULL, 'research', '{}', NULL, 'x', ?, ?, NULL)",
                      [run_id, NOW, status])


def _q(store):
    return {(r, lab): (p, q, qb, m) for r, lab, p, q, qb, m in store.con.execute(
        "SELECT run_id, label, p_value, q_value, q_by, m FROM hypothesis_q").fetchall()}


def _expected(p_values):
    return benjamini_hochberg(p_values), benjamini_yekutieli(p_values)


def test_nan_p_value_is_stored_as_null_and_left_out_of_the_family(store):
    _run(store, "R")
    for label, p in (("a", 0.001), ("b", 0.01), ("c", 0.02), ("d", math.nan), ("e", None)):
        store.log_hypothesis("R", label, "research", p, NOW)
    logged = dict(store.con.execute("SELECT label, p_value FROM hypothesis_log").fetchall())
    assert logged["d"] is None and logged["e"] is None and len(logged) == 5
    q = _q(store)
    bh, by = _expected([0.001, 0.01, 0.02])
    assert {k[1] for k in q} == {"a", "b", "c"} and {v[3] for v in q.values()} == {3}
    for label, b, y in zip("abc", bh, by):
        assert q[("R", label)][1:3] == pytest.approx((b, y), rel=1e-12)


def test_view_ignores_a_nan_written_directly(store):
    _run(store, "R")
    store.con.execute("INSERT INTO hypothesis_log VALUES ('R', 'x', 'research', 'NaN'::DOUBLE, ?)", [NOW])
    store.log_hypothesis("R", "y", "research", 0.02, NOW)
    assert _q(store) == {("R", "y"): (0.02, 0.02, 0.02, 1)}           # m = 1: q = p, c(1) = 1


@pytest.mark.parametrize("p", [-0.01, 1.0000001, math.inf])
def test_p_value_outside_the_unit_interval_is_refused(store, p):
    _run(store, "R")
    with pytest.raises(ValueError, match="outside"):
        store.log_hypothesis("R", "x", "research", p, NOW)
    assert store.con.execute("SELECT count(*) FROM hypothesis_log").fetchone()[0] == 0


def test_bounds_are_accepted(store):
    _run(store, "R")
    store.log_hypothesis("R", "zero", "research", 0.0, NOW)
    store.log_hypothesis("R", "one", "research", 1.0, NOW)
    q = _q(store)
    assert q[("R", "zero")][1] == 0.0 and q[("R", "one")][1] == 1.0


def test_relogging_a_label_replaces_it_and_does_not_grow_the_family(store):
    _run(store, "R")
    store.log_hypothesis("R", "x", "research", 0.5, NOW)
    store.log_hypothesis("R", "x", "research", 0.01, NOW)
    store.log_hypothesis("R", "y", "research", 0.04, NOW)
    q = _q(store)
    assert set(q) == {("R", "x"), ("R", "y")} and q[("R", "x")][0] == 0.01 and q[("R", "x")][3] == 2


def test_ties_get_equal_q_values_matching_python(store):
    _run(store, "R")
    ps = [0.01, 0.01, 0.01, 0.03, 0.03, 0.2]
    for i, p in enumerate(ps):
        store.log_hypothesis("R", f"t{i}", "research", p, NOW)
    q = _q(store)
    bh, by = _expected(ps)
    for i in range(len(ps)):
        assert q[("R", f"t{i}")][1:3] == pytest.approx((bh[i], by[i]), rel=1e-12)
    assert len({q[("R", f"t{i}")][1] for i in range(3)}) == 1


def test_only_ok_runs_are_in_the_family(store):
    for run_id, status in (("OK", "ok"), ("RUN", "running"), ("BAD", "failed"), ("INV", "invalid")):
        _run(store, run_id, status)
        store.log_hypothesis(run_id, "x", "research", 0.01, NOW)
    assert set(_q(store)) == {("OK", "x")}


def test_every_q_row_traces_to_one_logged_test_and_one_run(store):
    _run(store, "A")
    _run(store, "B")
    for run_id, ps in (("A", [0.001, 0.2]), ("B", [0.04, 0.04, 0.7])):
        for i, p in enumerate(ps):
            store.log_hypothesis(run_id, f"t{i}", "research", p, NOW)
    orphans = store.con.execute("""
        SELECT count(*) FROM hypothesis_q q
        LEFT JOIN hypothesis_log h USING (run_id, label) LEFT JOIN research_runs r USING (run_id)
        WHERE h.p_value IS DISTINCT FROM q.p_value OR r.status IS DISTINCT FROM 'ok' OR h.period <> q.period""").fetchone()[0]
    assert orphans == 0 and store.con.execute("SELECT count(*) FROM hypothesis_q").fetchone()[0] == 5


def test_invalidation_records_when_and_why_and_keeps_the_tests(store):
    _run(store, "R")
    store.log_hypothesis("R", "x", "research", 0.01, NOW)
    assert store.invalidate("R", "code defect: wrong join", NOW) == 1
    assert store.invalidate("missing", "nothing", NOW) == 0
    assert store.con.execute("SELECT * FROM run_invalidations").fetchall() == [("R", NOW, "code defect: wrong join")]
    assert store.con.execute("SELECT count(*) FROM hypothesis_log WHERE run_id = 'R'").fetchone()[0] == 1
    assert _q(store) == {}

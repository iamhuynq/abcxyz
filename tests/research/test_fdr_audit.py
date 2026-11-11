"""`quant audit fdr`: BY next to BH, registry linkage, and nothing written."""

from datetime import UTC, datetime

import pytest

from quant_research import registry
from quant_research.fdr_audit import render
from quant_research.results import ResultsStore
from tests.research.synthetic_research import build_research_db

NOW = datetime(2026, 11, 11, 9, tzinfo=UTC)


@pytest.fixture
def store(tmp_path):
    research = tmp_path / "research.duckdb"
    build_research_db(research)
    s = ResultsStore(tmp_path / "results.duckdb", research)
    yield s
    s.close()


def _log(store, run_id, pattern, p_values, kind="pattern", labels=None):
    store.con.execute("INSERT INTO research_runs VALUES (?, ?, ?, 'v1', 'research', '{}', NULL, 'x', ?, 'ok', NULL)",
                      [run_id, kind, pattern, NOW])
    for i, p in enumerate(p_values):
        label = labels[i] if labels else f"t{i}"
        store.con.execute("INSERT INTO hypothesis_log VALUES (?, ?, 'research', ?, ?)", [run_id, label, p, NOW])


def _registry(store):
    return (store.con.execute("SELECT * FROM hypotheses ORDER BY ALL").fetchall(),
            store.con.execute("SELECT * FROM hypothesis_transitions ORDER BY ALL").fetchall())


def test_report_flags_a_bh_finding_that_fails_by(store):
    # m = 2, c(m) = 1.5: p = 0.02 gives BH q = 0.04 (passes) and BY q = 0.06 (fails); p = 0.001 passes both
    _log(store, "R1", "order_imbalance_d10", [0.02, 0.9])
    _log(store, "R2", "momentum10_d10", [])
    before = _registry(store)
    text = render(store, NOW)
    assert _registry(store) == before
    assert "m = 2; c(m) = 1.500" in text
    assert "| R1 | t0 | pattern | order_imbalance_d10 | research | 0.02 | 0.04 | 0.06 | no |" in text
    assert "Tests with BH q < 0.05: 1; of these, BY q < 0.05: 0." in text
    row = next(line for line in text.splitlines() if line.startswith("| P3-C1 |"))
    assert row.endswith("| 1 | t0 | 0.04 | 0.06 | yes |")
    no_tests = next(line for line in text.splitlines() if line.startswith("| P3-C2 |"))
    assert no_tests.endswith("| 0 | - | - | - |  |")


def test_report_keeps_strong_findings(store):
    _log(store, "R1", "order_imbalance_d10", [0.0001, 0.9])
    text = render(store, NOW)
    assert "| R1 | t0 | pattern | order_imbalance_d10 | research | 0.0001 | 0.0002 | 0.0003 | yes |" in text
    row = next(line for line in text.splitlines() if line.startswith("| P3-C1 |"))
    assert row.endswith("| t0 | 0.0002 | 0.0003 |  |")


def test_an_interaction_hypothesis_uses_its_own_test_of_the_shared_scan(store):
    for hid in ("P7-IX-order_flow-breadth", "P7-IX-reversal_1w-liquidity"):
        registry.add(store.con, hid, "interaction", hid, "statement", NOW, "scan", pattern="interaction_scan")
    # m = 3, c(m) = 11/6: BH q = 0.0003 / 0.03 / 0.9, BY q = 0.00055 / 0.055 / 1
    _log(store, "IX", "interaction_scan", [0.0001, 0.02, 0.9], kind="interaction",
         labels=["ix_order_flow_breadth", "ix_reversal_1w_liquidity", "ix_other_dim"])
    text = render(store, NOW)
    strong = next(line for line in text.splitlines() if line.startswith("| P7-IX-order_flow-breadth |"))
    weak = next(line for line in text.splitlines() if line.startswith("| P7-IX-reversal_1w-liquidity |"))
    assert strong.endswith("| 1 | ix_order_flow_breadth | 0.0003 | 0.00055 |  |")
    assert weak.endswith("| 1 | ix_reversal_1w_liquidity | 0.03 | 0.055 | yes |")


def test_view_q_by_matches_python_by(store):
    import numpy as np

    from quant_research.stats import benjamini_hochberg, benjamini_yekutieli
    p = [float(x) for x in np.random.default_rng(3).uniform(size=40) ** 4]
    _log(store, "R1", "order_imbalance_d10", p)
    got = store.con.execute("SELECT label, q_value, q_by FROM hypothesis_q ORDER BY label").fetchall()
    labels = [f"t{i}" for i in range(len(p))]
    bh, by = dict(zip(labels, benjamini_hochberg(p))), dict(zip(labels, benjamini_yekutieli(p)))
    assert {lab: q for lab, q, _ in got} == pytest.approx(bh)
    assert {lab: qb for lab, _, qb in got} == pytest.approx(by)
    assert any(qb < 1 for _, _, qb in got) and any(qb > q for _, q, qb in got)


def test_a_hypothesis_without_a_pattern_is_linked_through_its_transitions(store):
    # P4-X2 has no pattern; its seeded transition names this run
    _log(store, "20261005T192527-cross-h1_leaders_to_followers-research", "h1_leaders_to_followers", [0.001],
         kind="cross")
    text = render(store, NOW)
    row = next(line for line in text.splitlines() if line.startswith("| P4-X2 |"))
    assert "| 1 | t0 | 0.001 | 0.001 |" in row


def test_tied_q_values_pick_one_row_deterministically(store):
    # equal p in two runs of the same pattern: equal q; the first run, then the first label, is shown
    _log(store, "R2", "order_imbalance_d10", [0.001, 0.001], labels=["b", "a"])
    _log(store, "R1", "order_imbalance_d10", [0.001, 0.9], labels=["z", "y"])
    text = render(store, NOW)
    row = next(line for line in text.splitlines() if line.startswith("| P3-C1 |"))
    assert "| 2 | z | " in row
    assert "global exploratory family" in text


def test_report_lists_tests_outside_the_family_and_rerun_of_each_invalid_run(store):
    _log(store, "20261001T100000-scan-features-research", None, [0.01, 0.2], kind="scan")
    _log(store, "20261002T100000-scan-features-research", None, [0.02, 0.3], kind="scan")
    _log(store, "20261001T100000-cross-h2-research", None, [0.04], kind="cross")
    _log(store, "20261003T100000-cross-h2-research", None, [0.05], kind="cross")
    store.con.execute("UPDATE research_runs SET created_at = created_at + INTERVAL 1 DAY "
                      "WHERE run_id LIKE '20261002%' OR run_id LIKE '20261003%'")
    store.con.execute("UPDATE research_runs SET status = 'invalid' WHERE run_id LIKE '20261001%'")
    store.con.execute("UPDATE research_runs SET status = 'failed' WHERE run_id = '20261003T100000-cross-h2-research'")
    store.con.execute("INSERT INTO hypothesis_log VALUES ('20261002T100000-scan-features-research', 'n', 'research', "
                      "NULL, ?)", [NOW])
    text = render(store, NOW).split("## 3.")[1]
    assert "| invalid | 3 | 0 |" in text and "| ok | 3 | 1 |" in text and "| failed | 1 | 0 |" in text
    assert "Invalidated runs with logged tests: 2; without a later `ok` run of the same name: 1." in text
    assert "| 20261001T100000-scan-features-research | 2 | 20261002T100000-scan-features-research |" in text
    assert "| 20261001T100000-cross-h2-research | 1 | none |" in text

"""Outcome-mask audit: the declared sensitivity rule, and an end-to-end run that leaves no trace in any real
results database."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from quant_research import mask_audit as ma
from quant_research.build import build
from quant_research.mask_audit_report import render, summary
from tests.research.synthetic import build_synthetic_warehouse

NOW = datetime(2027, 3, 1, 19, 0, tzinfo=UTC)


def tables(effect, q):
    return {"patterns": pd.DataFrame({"name": ["a", "b", "c", "d"], "horizon": ["h"] * 4, "effect": effect,
                                      "t": [1.0, 2.0, 3.0, 4.0], "q": q})}


def test_rule_flags_sign_changes_and_significance_flips_only():
    ref = tables([0.01, 0.01, -0.02, 0.03], [0.01, 0.20, 0.01, 0.01])
    other = tables([-0.01, 0.02, -0.03, 0.031], [0.01, 0.01, 0.02, 0.04])
    cmp = ma.compare({ma.REFERENCE: ref, "baseline": other}, "patterns", ("name", "horizon"))
    # a: sign flips; b: q crosses 0.05; c: same sign, both significant; d: unchanged
    assert list(cmp["baseline_sensitive"]) == [True, True, False, False]


def test_rule_ignores_missing_tests():
    ref = tables([0.01, None, 0.0, 0.03], [0.01, 0.01, 0.01, 0.01])
    other = tables([-0.01, -0.01, 0.01, None], [0.01, 0.01, 0.01, 0.01])        # c: zero reference effect
    cmp = ma.compare({ma.REFERENCE: ref, "baseline": other}, "patterns", ("name", "horizon"))
    assert list(cmp["baseline_sensitive"]) == [True, False, False, False]


@pytest.fixture(scope="module")
def research(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("maskaudit")
    build_synthetic_warehouse(tmp / "warehouse.duckdb")
    build(tmp / "warehouse.duckdb", tmp / "research.duckdb", NOW)
    return tmp / "research.duckdb"


def test_audit_runs_end_to_end_without_touching_real_results(research, monkeypatch):
    from quant_research import interactions
    from quant_research.mask_audit_report import extras
    monkeypatch.setattr(interactions, "MIN_STOCKS", 2)
    monkeypatch.setattr(interactions, "MIN_DATES", 5)
    configs = {"combined": {"min_adv_value": 0, "min_session_index": 0, "min_stocks_per_date": 2},
               "baseline": {"min_adv_value": 0, "min_session_index": 0, "min_stocks_per_date": 2,
                            "outcome_masks": ()}}
    before = sorted(p.name for p in research.parent.iterdir())
    results = ma.run_audit(research, None, NOW, log=lambda _m: None, configs=configs)
    assert sorted(p.name for p in research.parent.iterdir()) == before          # scratch files are gone
    rows = {n: r["sample"].set_index("target")["rows"] for n, r in results.items()}
    assert (rows["baseline"] >= rows["combined"]).all() and (rows["baseline"] > rows["combined"]).any()
    assert not results["combined"]["patterns"].empty
    ex = extras(research)
    prov = ex.pop("prov").iloc[0].to_dict()
    monkeypatch.setattr("quant_research.mask_audit_report.OTHERS", ["baseline"])
    text = render(results, ex, prov, NOW)
    assert "## 3. Mask-sensitive tests per configuration" in text
    assert set(summary(results)["analysis"]) >= {"patterns"}

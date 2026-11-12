"""Experiment JSON files and `quant ml run` (docs/ml-plan.md, PR #13): every field explicit, nothing defaulted."""

import argparse
import json

import pytest

from quant_research.ml import cli as ml_cli
from quant_research.ml import experiment_file
from quant_research.ml.contracts import ContractError
from quant_research.results import ResultsStore
from tests.research.ml_wf_fixture import build_wide
from tests.research.test_ml_runner import SPEC


def _doc() -> dict:
    s = SPEC
    return {"hypothesis": s.hypothesis,
            "dataset": {"features": [{"table": f.table, "column": f.column, "source_version": f.source_version}
                                     for f in s.dataset.features],
                        "label": {"column": s.dataset.label.column, "horizon": s.dataset.label.horizon},
                        "period_start": "2021-01-04", "period_end": "2022-12-30", "universe": s.dataset.universe},
            "split": {"kind": "expanding", "train_min_sessions": 120, "test_sessions": 60, "step_sessions": 60,
                      "embargo_sessions": 5, "first_date": "2021-01-04", "last_date": "2022-12-30",
                      "train_sessions": None, "min_train_rows": 1000},
            "model": "ridge", "params": {"alpha": 1.0}, "seed": 0,
            "baseline": {"factor": "volume_surge", "source_version": s.baseline.source_version, "field": "rank_pct"},
            "primary_test": "incremental_rank_ic_newey_west"}


def test_a_complete_file_gives_the_same_experiment():
    spec = experiment_file.parse(_doc())
    assert spec == SPEC and spec.experiment_id == SPEC.experiment_id


@pytest.mark.parametrize("edit, why", [
    (lambda d: d.pop("seed"), "experiment must have exactly"),
    (lambda d: d.update(extra=1), "experiment must have exactly"),
    (lambda d: d["dataset"].pop("universe"), "dataset must have exactly"),
    (lambda d: d["split"].pop("train_sessions"), "split must have exactly"),
    (lambda d: d["dataset"]["features"][0].pop("source_version"), "feature must have exactly"),
    (lambda d: d["dataset"]["features"][0].update(source_version="old"), "source version|not the current|version"),
    (lambda d: d["dataset"].update(period_end="2024-03-01"), "research cutoff"),
    (lambda d: d["split"].update(first_date="03/01/2021"), "ISO date"),
    (lambda d: d.update(model="mean"), "approved family"),
    (lambda d: d["baseline"].update(factor="nope"), "not a Factor Engine factor"),
    (lambda d: d["dataset"].update(features={}), "must be a list"),
])
def test_incomplete_or_invalid_files_are_refused(edit, why):
    d = _doc()
    edit(d)
    with pytest.raises(ContractError, match=why):
        experiment_file.parse(d)


def test_unreadable_file_is_refused(tmp_path):
    (tmp_path / "e.json").write_text("{nope")
    with pytest.raises(ContractError, match="unreadable"):
        experiment_file.load(tmp_path / "e.json")


def test_cli_runs_once_then_refuses(tmp_path, capsys):
    research = build_wide(tmp_path)
    (tmp_path / "e.json").write_text(json.dumps(_doc()))
    args = argparse.Namespace(experiment=tmp_path / "e.json", runs_dir=tmp_path / "runs")

    def factory():
        return ResultsStore(tmp_path / "results.duckdb", research)
    assert ml_cli.run(args, factory, tmp_path) == 0
    out = capsys.readouterr().out
    assert "status: ok" in out and SPEC.experiment_id in out and "lag 4" in out
    assert ml_cli.run(args, factory, tmp_path) == 2
    assert "Refused (nothing logged)" in capsys.readouterr().err
    (tmp_path / "bad.json").write_text(json.dumps({**_doc(), "seed": -1}))
    assert ml_cli.run(argparse.Namespace(experiment=tmp_path / "bad.json", runs_dir=None), factory, tmp_path) == 2
    with factory() as s:
        assert s.con.execute("SELECT count(*) FROM research_runs").fetchone()[0] == 1   # none for the bad file

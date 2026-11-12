"""Walk-forward runner (docs/ml-plan.md, PR #13): preflight before anything is logged, purged folds checked on their
exact frames, test labels never used before prediction, one result per experiment and dataset, crash safety,
atomic and reproducible run artifacts."""

import json
from dataclasses import replace
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
import pytest

from quant_research.ml import artifacts, evaluation, runner, training
from quant_research.ml.contracts import (
    BaselineSpec,
    DatasetSpec,
    ExperimentSpec,
    FeatureSpec,
    LabelSpec,
    SplitSpec,
)
from quant_research.results import ResearchParams, ResultsStore, stored_params
from tests.research.ml_fixture import reseal
from tests.research.ml_wf_fixture import build_wide

F = FeatureSpec.current
START, END = date(2021, 1, 4), date(2022, 12, 30)
NOW = datetime(2026, 11, 12, 10, 0, tzinfo=UTC)
LABEL = "ml_incremental_rank_ic_fwd_ret_exec_5d"


def _spec(**split) -> ExperimentSpec:
    ds = DatasetSpec((F("stock_features", "return_5d"), F("stock_features", "volume_ratio_20"),
                      F("stock_features", "volatility_20")), LabelSpec("fwd_ret_exec_5d", 5), START, END)
    sp = SplitSpec(**({"kind": "expanding", "train_min_sessions": 120, "test_sessions": 60, "step_sessions": 60,
                       "embargo_sessions": 5, "first_date": START, "last_date": END, "train_sessions": None,
                       "min_train_rows": 1000} | split))
    return ExperimentSpec("ridge on three features beats volume surge", ds, sp, "ridge", 0,
                          BaselineSpec.current("volume_surge"), "incremental_rank_ic_newey_west", {"alpha": 1.0})


SPEC = _spec()


@pytest.fixture(scope="module")
def research(tmp_path_factory):
    return build_wide(tmp_path_factory.mktemp("mlwf"))


@pytest.fixture
def store(research, tmp_path):
    with ResultsStore(tmp_path / "results.duckdb", research) as s:
        yield s


def _rows(store, sql="SELECT run_id, status FROM research_runs ORDER BY created_at"):
    return store.con.execute(sql).fetchall()


def _prepared(store, spec=SPEC):
    result = runner.build_dataset(store.con, spec.dataset, store.feature_build_id(), schema="rs.")
    ds = training.open_dataset(result)
    baseline = runner.load_baseline(store.con, spec, ds.data)
    return result, ds, baseline, runner.load_calendar(store.con, spec)


def test_a_run_logs_one_test_publishes_every_fold_and_finishes(store, tmp_path):
    run_id, primary = runner.run_experiment(store, SPEC, tmp_path / "runs", NOW)
    assert _rows(store) == [(run_id, "ok")]
    hyp = _rows(store, "SELECT run_id, label, period, p_value FROM hypothesis_log")
    assert hyp == [(run_id, LABEL, "research", primary["p"])] and 0 <= primary["p"] <= 1
    assert _rows(store, "SELECT count(*) FROM hypothesis_q")[0][0] == 1
    kind, name, version, params = _rows(store, "SELECT kind, pattern_name, pattern_version, params "
                                               "FROM research_runs")[0]
    cfg = json.loads(params)["config"]
    assert (kind, name) == ("ml", SPEC.experiment_id) and version == cfg["dataset_fingerprint"]
    assert cfg["experiment_id"] == SPEC.experiment_id and cfg["min_stocks"] == 30 and cfg["min_dates"] == 60
    manifest = runner.verify_run(tmp_path / "runs" / run_id)
    assert "status" not in manifest and manifest["primary"] == json.loads(json.dumps(primary))
    assert manifest["dataset_verifications"] == 1 + 2 * len(manifest["folds"]) and len(manifest["folds"]) == 6
    keys = set()
    for m in manifest["folds"]:
        out = tmp_path / "runs" / run_id / "folds" / f"{m['fold']:03d}"
        pred = pd.read_parquet(out / "predictions.parquet")
        assert m["valid_dates"] >= 60 and m["purged_rows"] > 0 and m["baseline_sign"] in (1, -1)
        assert set(pred["fold"]) == {m["fold"]} and set(pred["model_id"]) == {m["model_id"]}
        assert pd.to_datetime(pred["date"]).dt.date.between(date.fromisoformat(m["test_start"]),
                                                             date.fromisoformat(m["test_end"])).all()
        k = set(zip(pred["date"], pred["symbol"], strict=True))
        assert not keys & k
        keys |= k
        trained = artifacts.load_model(out / "model")
        assert trained.model_id == m["model_id"]
        assert trained.metadata["training"]["rows"] == m["train_rows"]             # fitted on its training rows
        assert trained.metadata["training"]["keys_hash"] == m["train_keys_hash"]
    daily = pd.concat(pd.read_parquet(tmp_path / "runs" / run_id / "folds" / f"{m['fold']:03d}" / "daily_ic.parquet")
                      for m in manifest["folds"])
    assert evaluation.primary_test(daily.reset_index(drop=True), 5)["p"] == primary["p"]


def test_folds_are_purged_and_checked_on_their_exact_frames(store, monkeypatch):
    _result, ds, baseline, cal = _prepared(store)
    seen = []
    real = runner.check_fold
    monkeypatch.setattr(runner, "check_fold", lambda tr, te, f, sp: (seen.append((tr, te, f)), real(tr, te, f, sp)))
    plans = runner.preflight(ds, baseline, cal, SPEC.split)
    assert len(seen) == len(plans) == 6
    d = pd.to_datetime(ds.data["date"]).dt.date
    for p, (tr, te, f) in zip(plans, seen, strict=True):
        pd.testing.assert_frame_equal(tr, ds.data.iloc[p.train_rows][runner.INTERVAL])
        pd.testing.assert_frame_equal(te, ds.data.iloc[p.test_rows][runner.INTERVAL])
        train = ds.data.iloc[p.train_rows]
        assert (pd.to_datetime(train["label_end"]).dt.date < f.embargo_start).all()
        cand = ds.data[(d >= f.train_start) & (d <= f.train_end)]
        purged = cand[pd.to_datetime(cand["label_end"]).dt.date >= f.embargo_start]
        assert len(purged) == p.candidate_train_rows - len(p.train_rows) > 0
        assert (np.diff(p.train_rows) > 0).all()


def test_test_labels_are_not_used_before_prediction(store):
    result, ds, baseline, cal = _prepared(store)
    plans = runner.preflight(ds, baseline, cal, SPEC.split)
    k = 2
    f = plans[k].fold
    data = ds.data.copy()
    later = pd.to_datetime(data["date"]).dt.date >= f.test_start
    data.loc[later, "fwd_ret_exec_5d"] = -data.loc[later, "fwd_ret_exec_5d"] * 3      # this fold's test labels and after
    other = reseal(result, data)
    plans2 = runner.preflight(training.open_dataset(other), baseline, cal, SPEC.split)
    for a, b in zip(plans[:k + 1], plans2[:k + 1], strict=True):
        assert (a.sign, a.train_mean_ic) == (b.sign, b.train_mean_ic)
        np.testing.assert_array_equal(a.train_rows, b.train_rows)
    model = runner.ModelSpec("ridge", {"alpha": 1.0}, 0)
    t1, t2 = (training.fit_model(r, plans[k].train_rows, model) for r in (result, other))
    assert t1.fitted == t2.fitted
    p1 = training.predict_model(t1, result, plans[k].test_rows)["prediction"]
    p2 = training.predict_model(t2, other, plans[k].test_rows)["prediction"]
    np.testing.assert_array_equal(p1, p2)


def test_the_baseline_is_the_factor_score_of_the_same_keys(store):
    _, ds, baseline, _ = _prepared(store)
    sf = store.con.execute("SELECT date, symbol, rank_pct FROM rs.stock_factors WHERE factor = 'volume_surge'").df()
    sf["date"] = pd.to_datetime(sf["date"]).dt.date
    look = {(d, s): v for d, s, v in sf.itertuples(index=False)}
    keys = zip(pd.to_datetime(ds.data["date"]).dt.date, ds.data["symbol"], strict=True)
    expect = np.array([look.get(k, np.nan) for k in keys], dtype=np.float64)
    np.testing.assert_array_equal(baseline, expect)
    assert np.isnan(baseline).sum() < len(baseline) // 10


@pytest.mark.parametrize("change, why", [
    ({"step_sessions": 50}, "test windows overlap"),
    ({"test_sessions": 59, "step_sessions": 59}, "test_sessions 59 < 60"),
    ({"min_train_rows": 10**6}, "training rows"),
])
def test_preflight_refusals_log_nothing(store, tmp_path, change, why):
    with pytest.raises(runner.ExperimentRefused, match=why):
        runner.run_experiment(store, _spec(**change), tmp_path / "runs", NOW)
    assert _rows(store) == [] and _rows(store, "SELECT * FROM hypothesis_log") == []
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").iterdir())


def test_threshold_and_sign_refusals_log_nothing(store, tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "MIN_STOCKS", 46)                       # 45 stocks: no date qualifies
    with pytest.raises(runner.ExperimentRefused, match="test dates with >= 46 stocks"):
        runner.run_experiment(store, SPEC, tmp_path / "runs", NOW)
    monkeypatch.undo()
    monkeypatch.setattr(evaluation, "daily_baseline_ic", lambda train: pd.Series([0.1, -0.1] * 40, dtype=float))
    with pytest.raises(runner.ExperimentRefused, match="no sign is chosen"):
        runner.run_experiment(store, SPEC, tmp_path / "runs", NOW)
    assert _rows(store) == [] and _rows(store, "SELECT * FROM hypothesis_log") == []


def test_one_result_per_experiment_and_dataset(store, tmp_path):
    run_id, _ = runner.run_experiment(store, SPEC, tmp_path / "runs", NOW)
    with pytest.raises(runner.ExperimentRefused, match="one result per experiment"):
        runner.run_experiment(store, SPEC, tmp_path / "runs", NOW.replace(hour=11))
    assert len(_rows(store)) == 1
    other, _ = runner.run_experiment(store, replace(SPEC, seed=1), tmp_path / "runs", NOW.replace(hour=12))
    assert other != run_id                                                   # another experiment: its own test
    store.invalidate(run_id, "test: established defect")
    again, _ = runner.run_experiment(store, SPEC, tmp_path / "runs", NOW.replace(hour=13))
    assert again not in (run_id, other)
    assert _rows(store, "SELECT count(*) FROM hypothesis_q")[0][0] == 2      # the invalid run is outside FDR


def test_the_label_is_declared_before_the_result(store, tmp_path, monkeypatch):
    seen = []
    real = evaluation.primary_test

    def spy(daily, horizon):
        seen.append(_rows(store, "SELECT label, p_value FROM hypothesis_log"))
        seen.append(_rows(store))
        return real(daily, horizon)
    monkeypatch.setattr(evaluation, "primary_test", spy)
    run_id, _ = runner.run_experiment(store, SPEC, tmp_path / "runs", NOW)
    assert seen == [[(LABEL, None)], [(run_id, "running")]]


def test_a_technical_failure_before_a_result_allows_a_retry(research, tmp_path, monkeypatch):
    with ResultsStore(tmp_path / "results.duckdb", research) as s:
        monkeypatch.setattr(training, "fit_model", lambda *a: (_ for _ in ()).throw(MemoryError("out of memory")))
        with pytest.raises(MemoryError):
            runner.run_experiment(s, SPEC, tmp_path / "runs", NOW)
        monkeypatch.undo()
    with ResultsStore(tmp_path / "results.duckdb", research) as s:
        (failed, status), = _rows(s)
        assert status == "failed" and _rows(s, "SELECT p_value FROM hypothesis_log") == [(None,)]
        assert not (tmp_path / "runs" / failed).exists() and not list((tmp_path / "runs").glob(".*tmp-*"))
        run_id, _ = runner.run_experiment(s, SPEC, tmp_path / "runs", NOW.replace(hour=11))
        assert _rows(s, "SELECT run_id FROM hypothesis_q") == [(run_id,)]   # one test in the FDR family


def test_a_crash_after_publishing_never_leaves_an_ok_run_and_blocks_a_rerun(research, tmp_path, monkeypatch):
    with ResultsStore(tmp_path / "results.duckdb", research) as s:
        monkeypatch.setattr(ResultsStore, "finish_run", lambda self, rid: (_ for _ in ()).throw(OSError("crash")))
        with pytest.raises(OSError):
            runner.run_experiment(s, SPEC, tmp_path / "runs", NOW)
        monkeypatch.undo()
    with ResultsStore(tmp_path / "results.duckdb", research) as s:
        (run_id, status), = _rows(s)
        assert status == "failed" and (tmp_path / "runs" / run_id / "manifest.json").exists()
        assert _rows(s, "SELECT p_value FROM hypothesis_log") == [(None,)]   # the p-value was rolled back
        with pytest.raises(runner.ExperimentRefused, match="one result per experiment"):
            runner.run_experiment(s, SPEC, tmp_path / "runs", NOW.replace(hour=11))
        assert _rows(s, "SELECT count(*) FROM hypothesis_q")[0][0] == 0


def test_a_killed_run_left_running_blocks_a_rerun(store, tmp_path, monkeypatch):
    run_id = "20260101T000000-ml-x-research"
    _, ds, _, _ = _prepared(store)
    store.start_run(run_id, "ml", "research", ResearchParams(), NOW, (SPEC.experiment_id, ds.fingerprint), {})
    store._open_runs.discard(run_id)                                         # as if the process had been killed
    with pytest.raises(runner.ExperimentRefused, match="running"):
        runner.run_experiment(store, SPEC, tmp_path / "runs", NOW)


def test_runs_are_reproducible(research, tmp_path):
    out = []
    for i in range(2):
        with ResultsStore(tmp_path / f"r{i}.duckdb", research) as s:
            run_id, primary = runner.run_experiment(s, SPEC, tmp_path / f"runs{i}", NOW)
        m = runner.verify_run(tmp_path / f"runs{i}" / run_id)
        preds = [pd.read_parquet(tmp_path / f"runs{i}" / run_id / "folds" / f"{f['fold']:03d}" / "predictions.parquet")
                 for f in m["folds"]]
        drop = ("fold_seconds",)
        out.append(({k: v for k, v in m.items() if k not in drop}, preds, primary))
    assert out[0][0] == out[1][0] and out[0][2] == out[1][2]
    for a, b in zip(out[0][1], out[1][1], strict=True):
        pd.testing.assert_frame_equal(a, b)


def test_start_run_without_config_is_unchanged(store):
    store.start_run("a", "pattern", "research", ResearchParams(), NOW)
    store.start_run("b", "ml", "research", ResearchParams(), NOW, ("e", "f"), {"x": 1})
    a, b = (json.loads(p) for (p,) in _rows(store, "SELECT params FROM research_runs ORDER BY run_id"))
    base = stored_params(ResearchParams())
    assert {k: v for k, v in a.items() if k != "provenance"} == json.loads(json.dumps(base, default=str))
    assert b["config"] == {"x": 1} and a["provenance"]["config_hash"] != b["provenance"]["config_hash"]
    with pytest.raises(ValueError):
        store.start_run("c", "ml", "research", ResearchParams(), NOW, None, [1])
    for rid in ("a", "b"):
        store.finish_run(rid)


def test_a_test_row_in_two_folds_is_refused_even_if_the_fold_generator_errs(store, monkeypatch):
    _result, ds, baseline, cal = _prepared(store)
    real = runner.folds
    monkeypatch.setattr(runner, "folds", lambda c, sp: [real(c, sp)[0]] * 2)
    with pytest.raises(runner.ExperimentRefused, match="already tested by an earlier fold"):
        runner.preflight(ds, baseline, cal, SPEC.split)


def test_a_logged_p_value_blocks_a_rerun(store, tmp_path):
    _, ds, _, _ = _prepared(store)
    rid = "20260101T000000-ml-y-research"
    store.start_run(rid, "ml", "research", ResearchParams(), NOW, (SPEC.experiment_id, ds.fingerprint), {})
    store.log_hypothesis(rid, LABEL, "research", 0.5, NOW)
    store.con.execute("UPDATE research_runs SET status = 'failed' WHERE run_id = ?", [rid])
    store._open_runs.discard(rid)
    with pytest.raises(runner.ExperimentRefused, match="one result per experiment"):
        runner.run_experiment(store, SPEC, tmp_path / "runs", NOW)


def test_verify_run_detects_a_damaged_run(store, tmp_path):
    run_id, _ = runner.run_experiment(store, SPEC, tmp_path / "runs", NOW)
    path = tmp_path / "runs" / run_id
    model = path / "folds" / "001" / "model" / "model.json"
    doc = json.loads(model.read_text())
    doc["fitted"]["intercept"] = (1.0).hex()
    model.write_text(json.dumps(doc))
    with pytest.raises(runner.ContractError, match="model_id"):
        runner.verify_run(path)
    model.unlink()
    with pytest.raises(runner.ContractError):
        runner.verify_run(path)
    pred = path / "folds" / "002" / "predictions.parquet"
    pd.read_parquet(pred).iloc[:-1].to_parquet(pred, index=False)
    m = json.loads((path / "manifest.json").read_text())
    m["folds"] = m["folds"][2:]
    (path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(runner.ContractError, match="predictions are incomplete"):
        runner.verify_run(path)


def test_a_fold_left_below_60_dates_after_prediction_fails_the_run(research, tmp_path, monkeypatch):
    real = training.predict_model

    def flat_first_days(trained, dataset, rows):                            # constant predictions on 10 dates
        out = real(trained, dataset, rows)
        first = sorted(out["date"].unique())[:10]
        out.loc[out["date"].isin(first), "prediction"] = 0.0
        return out
    monkeypatch.setattr(training, "predict_model", flat_first_days)
    with ResultsStore(tmp_path / "results.duckdb", research) as s:
        with pytest.raises(runner.ContractError, match="valid dates after prediction"):
            runner.run_experiment(s, SPEC, tmp_path / "runs", NOW)
    with ResultsStore(tmp_path / "results.duckdb", research) as s:
        (run_id, status), = _rows(s)
        assert status == "failed" and _rows(s, "SELECT p_value FROM hypothesis_log") == [(None,)]
        assert not (tmp_path / "runs" / run_id).exists()

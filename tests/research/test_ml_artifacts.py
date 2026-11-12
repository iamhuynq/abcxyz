"""Model artifacts (docs/ml-plan.md, PR #12): canonical JSON only, exact round trip, model_id recomputed on load,
atomic publish that never overwrites, and an end-to-end run on the PR #11 synthetic dataset."""

import json

import numpy as np
import pytest

from quant_research.ml import artifacts, dataset, training
from quant_research.ml.contracts import ContractError, ModelSpec
from tests.research.ml_fixture import build_dataset

SPECS = (ModelSpec("mean"), ModelSpec("ridge", {"alpha": 0.5}),
         ModelSpec("elasticnet", {"alpha": 0.0005, "l1_wt": 0.5}, seed=3))


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    return build_dataset(tmp_path_factory.mktemp("mlart"))


@pytest.fixture(scope="module")
def trained(ds):
    return training.fit_model(ds, list(range(len(ds.data) // 2)), SPECS[1])


def _edit(path, change):
    doc = json.loads((path / "model.json").read_text())
    change(doc)
    (path / "model.json").write_text(json.dumps(doc))


def test_end_to_end_train_predict_save_load_predict(ds, tmp_path):
    n = len(ds.data)
    train, test = list(range(n // 2)), list(range(n // 2 + 10, n))
    loaded_ds = dataset.read_dataset(dataset.write_dataset(ds, tmp_path / "ds"))
    for spec in SPECS:
        t = training.fit_model(ds, train, spec)
        before = training.predict_model(t, ds, test)
        back = artifacts.load_model(artifacts.save_model(t, tmp_path / spec.model))
        assert back.model_id == t.model_id and back.fitted == t.fitted and back.spec == t.spec
        after = training.predict_model(back, loaded_ds, test)
        assert after.equals(before)                                        # exact: numbers stored as float.hex
        assert after["model_id"].iloc[0] == t.model_id


def test_artifact_is_json_only_with_a_versioned_schema(trained, tmp_path):
    path = artifacts.save_model(trained, tmp_path / "m")
    assert sorted(p.name for p in path.iterdir()) == ["model.json"]
    doc = json.loads((path / "model.json").read_text())
    assert doc["artifact_schema_version"] == "model/1" and set(doc) == artifacts.FIELDS
    assert all(isinstance(v, str) for v in doc["fitted"]["coef"])          # float.hex
    assert doc["metadata"]["dataset_fingerprint"] == trained.metadata["dataset_fingerprint"]
    import inspect
    src = inspect.getsource(artifacts) + inspect.getsource(training)
    assert "pickle" not in src.replace("no pickle", "") and "joblib" not in src.replace("no joblib", "")


def test_created_at_is_not_part_of_the_identity(trained, tmp_path):
    path = artifacts.save_model(trained, tmp_path / "m")
    _edit(path, lambda d: d.update(created_at="2000-01-01T00:00:00+00:00"))
    assert artifacts.load_model(path).model_id == trained.model_id


@pytest.mark.parametrize("change, why", [
    (lambda d: d["metadata"].update(environment="python 0"), "model_id"),
    (lambda d: d["metadata"].update(label="fwd_ret_exec_10d"), "label differs"),
    (lambda d: d["metadata"]["dataset_provenance"].update(kept_rows=10**6), "dataset_fingerprint"),
    (lambda d: d["metadata"].update(model_code_hash="z" * 16), "model_code_hash"),
    (lambda d: d["metadata"]["training"].update(rows=1), "model_id"),
    (lambda d: d["fitted"]["coef"].__setitem__(0, (1.5).hex()), "model_id"),
    (lambda d: d["fitted"].update(intercept=(0.25).hex()), "model_id"),
    (lambda d: d["spec"].update(seed=9), "model_id"),
    (lambda d: d.update(model_id="0" * 16), "model_id"),
    (lambda d: d["spec"]["params"].update(alpha=-1), "alpha must be > 0"),
    (lambda d: d["spec"].update(model="mean", params={}), "fitted state is ridge"),
    (lambda d: d.update(artifact_schema_version="model/2"), "unsupported artifact schema"),
    (lambda d: d.pop("created_at"), "exactly the fields"),
    (lambda d: d.update(extra=1), "exactly the fields"),
    (lambda d: d["fitted"].update(coef=d["fitted"]["coef"][:2]), "inconsistent shapes"),
    (lambda d: d["fitted"].update(means=d["fitted"]["means"][:2], scales=d["fitted"]["scales"][:2],
                                   coef=d["fitted"]["coef"][:2]), "3 features for 2 coefficients"),
    (lambda d: d["fitted"].update(intercept="nan"), "finite"),
    (lambda d: d["fitted"].update(intercept=0.5), "float.hex"),
    (lambda d: d["metadata"].pop("features"), "malformed metadata: keys"),
    (lambda d: d["fitted"].update(model="lasso"), "not an approved family"),
])
def test_edited_artifacts_are_refused(trained, tmp_path, change, why):
    path = artifacts.save_model(trained, tmp_path / "m")
    _edit(path, change)
    with pytest.raises(ContractError, match=why):
        artifacts.load_model(path)


def test_malformed_or_missing_files_are_refused(trained, tmp_path):
    path = artifacts.save_model(trained, tmp_path / "m")
    (path / "model.json").write_text("{not json")
    with pytest.raises(ContractError, match="unreadable"):
        artifacts.load_model(path)
    (path / "model.json").write_text("[]")
    with pytest.raises(ContractError, match="exactly the fields"):
        artifacts.load_model(path)
    with pytest.raises(ContractError, match="unreadable"):
        artifacts.load_model(tmp_path / "absent")


def test_never_overwrites_and_a_failed_publish_leaves_nothing(trained, tmp_path, monkeypatch):
    path = artifacts.save_model(trained, tmp_path / "m")
    original = (path / "model.json").read_text()
    with pytest.raises(FileExistsError, match="never overwritten"):
        artifacts.save_model(trained, path)
    assert (path / "model.json").read_text() == original
    monkeypatch.setattr(artifacts, "load_model", lambda p: (_ for _ in ()).throw(ContractError("verify failed")))
    with pytest.raises(ContractError, match="verify failed"):
        artifacts.save_model(trained, tmp_path / "m2")
    monkeypatch.undo()
    assert not (tmp_path / "m2").exists() and not list(tmp_path.glob(".m2.tmp-*"))
    artifacts.save_model(trained, tmp_path / "m2")                         # the lock was released


def test_a_tampered_trained_model_is_not_saved(trained, tmp_path):
    from dataclasses import replace
    with pytest.raises(ContractError, match="model_id"):
        artifacts.save_model(replace(trained, model_id="0" * 16), tmp_path / "m")
    assert not (tmp_path / "m").exists()


def test_save_load_keeps_predictions_identical_for_any_batch(ds, trained, tmp_path):
    back = artifacts.load_model(artifacts.save_model(trained, tmp_path / "m"))
    rows = list(range(len(ds.data)))
    whole = training.predict_model(back, ds, rows).set_index("row")["prediction"]
    single = [training.predict_model(back, ds, [i])["prediction"].iloc[0] for i in rows[::17]]
    np.testing.assert_array_equal(whole.loc[rows[::17]].to_numpy(), single)   # a row's value ignores its batch

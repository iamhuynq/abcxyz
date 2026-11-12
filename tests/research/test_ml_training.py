"""Fit and predict on explicit rows of one verified dataset (docs/ml-plan.md, PR #12): index checks, dataset checks,
leakage and adversarial cases, order and key preservation, reproducibility across processes."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from quant_research.ml import dataset, models, training
from quant_research.ml.contracts import ContractError, ModelSpec
from tests.research.ml_fixture import build_dataset, reseal

RIDGE = ModelSpec("ridge", {"alpha": 0.5})
SPECS = (ModelSpec("mean"), RIDGE, ModelSpec("elasticnet", {"alpha": 0.0005, "l1_wt": 0.5}))


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    return build_dataset(tmp_path_factory.mktemp("mltrain"))


def _split(ds):
    n = len(ds.data)
    return list(range(n // 2)), list(range(n // 2 + 10, n))


def test_fit_and_predict_every_model_keep_rows_keys_and_order(ds):
    train, test = _split(ds)
    shuffled = list(reversed(test))[::2] + test[::2]
    for spec in SPECS:
        t = training.fit_model(ds, train, spec)
        out = training.predict_model(t, ds, shuffled)
        assert out["row"].tolist() == shuffled
        sel = ds.data.iloc[shuffled]
        assert out["date"].tolist() == sel["date"].tolist() and out["symbol"].tolist() == sel["symbol"].tolist()
        assert out["label"].tolist() == sel["fwd_ret_exec_5d"].tolist()
        assert set(out["label_name"]) == {"fwd_ret_exec_5d"} and set(out["model_id"]) == {t.model_id}
        assert set(out["dataset_fingerprint"]) == {ds.fingerprint}
        direct = models.predict(t.fitted, sel[list(ds.data.columns[2:5])].to_numpy(dtype=np.float64))
        np.testing.assert_array_equal(out["prediction"].to_numpy(), direct)
        assert list(out.columns) == ["row", "date", "symbol", "label_name", "prediction", "label", "model_id",
                                     "dataset_fingerprint"]


def test_training_metadata_comes_from_the_selected_rows(ds):
    rows = list(range(20, 60))
    t = training.fit_model(ds, rows, RIDGE)
    m, sel = t.metadata, ds.data.iloc[rows]
    tr = m["training"]
    assert tr["rows"] == 40 and tr["first_date"] == str(sel["date"].min()) and tr["last_date"] == str(sel["date"].max())
    assert tr["last_label_end"] == str(sel["label_end"].max())
    assert tr["first_label_start"] == str(sel["label_start"].min())
    assert m["dataset_fingerprint"] == ds.fingerprint and m["label"] == "fwd_ret_exec_5d" and m["horizon"] == 5
    assert [f for f, _v in m["features"]] == list(ds.data.columns[2:5])
    assert dict(m["features"]) == dict(ds.metadata["feature_versions"])
    assert m["dataset_provenance"]["content_hash"] == ds.metadata["content_hash"]
    assert tr["keys_hash"] != training.fit_model(ds, list(range(21, 61)), RIDGE).metadata["training"]["keys_hash"]
    assert m["solver"] == {} and training.fit_model(ds, rows, SPECS[2]).metadata["solver"] == models.SOLVER["elasticnet"]


@pytest.mark.parametrize("idx, why", [
    ([], "non-empty"), ([0, 0, 1], "duplicates"), ([-1, 2], "within"), ([0, 10**6], "within"),
    ([0.0, 1.0], "integers"), ([True, False], "integers"), ([0, True], "integers"), (np.array([[0, 1]]), "1-D"),
    (np.array([0.0, 1.0]), "integer"), (np.array([True, False]), "integer"), ("01", "sequence"), (5, "sequence"),
    ([3, 1, 2], "strictly increasing"),
])
def test_bad_train_indices_are_refused(ds, idx, why):
    with pytest.raises(ContractError, match=why):
        training.fit_model(ds, idx, RIDGE)


def test_prediction_indices_may_be_unordered_but_are_otherwise_checked(ds):
    t = training.fit_model(ds, list(range(50)), RIDGE)
    assert len(training.predict_model(t, ds, np.array([9, 3, 7]))) == 3
    for idx in ([], [1, 1], [-1], [len(ds.data)], [1.0], [True]):
        with pytest.raises(ContractError):
            training.predict_model(t, ds, idx)


def test_prediction_never_reads_labels_and_fit_never_reads_other_rows(ds):
    train, test = _split(ds)
    t = training.fit_model(ds, train, RIDGE)
    base = training.predict_model(t, ds, test)["prediction"]
    data = ds.data.copy()
    data.loc[test, "fwd_ret_exec_5d"] = 99.0                               # prediction rows' labels
    other = reseal(ds, data)
    t2 = training.fit_model(other, train, RIDGE)
    assert t2.fitted == t.fitted                                           # same fit
    with pytest.raises(ContractError, match="predicts only on its own dataset"):
        training.predict_model(t, other, test)
    np.testing.assert_array_equal(training.predict_model(t2, other, test)["prediction"], base)
    data = ds.data.copy()
    out_rows = [i for i in range(len(data)) if i not in set(train)]
    data.loc[out_rows, data.columns[2:6]] = data.loc[out_rows, data.columns[2:6]] * 7 + 3   # features and labels
    for spec in SPECS:
        assert training.fit_model(reseal(ds, data), train, spec).fitted == training.fit_model(ds, train, spec).fitted


def test_standardization_statistics_are_those_of_the_training_rows(ds):
    train, _ = _split(ds)
    t = training.fit_model(ds, train, RIDGE)
    X = ds.data.iloc[train][list(ds.data.columns[2:5])].to_numpy(dtype=np.float64)
    X = np.ascontiguousarray(X)
    np.testing.assert_array_equal(t.fitted.means, X.mean(0))
    np.testing.assert_array_equal(t.fitted.scales, X.std(0, ddof=0))
    full = np.ascontiguousarray(ds.data[list(ds.data.columns[2:5])].to_numpy(dtype=np.float64))
    assert not np.allclose(t.fitted.means, full.mean(0), rtol=1e-6, atol=0)   # not the full-dataset statistics


def test_tampered_or_foreign_datasets_are_refused(ds):
    train, test = _split(ds)
    t = training.fit_model(ds, train, RIDGE)
    data = ds.data.copy()
    data.iloc[0, 2] = 0.5
    from dataclasses import replace
    stale = replace(ds, data=data)                                         # content changed, fingerprint stale
    for call in (lambda: training.fit_model(stale, train, RIDGE), lambda: training.predict_model(t, stale, test)):
        with pytest.raises(ContractError, match="content hash"):
            call()
    with pytest.raises(ContractError, match="fingerprint"):
        training.fit_model(replace(ds, fingerprint="0" * 16), train, RIDGE)
    with pytest.raises(ContractError, match="DatasetResult or the"):
        training.fit_model(ds.data, train, RIDGE)


def test_self_consistent_but_malformed_datasets_are_refused(ds):
    train = list(range(40))
    d = ds.data
    cols = list(d.columns)
    swapped = cols[:2] + [cols[3], cols[2]] + cols[4:]                     # feature order changed
    renamed = d.rename(columns={cols[2]: "stock_features.return_10d"})
    with_label = d.assign(fwd_ret_exec_5d_copy=d["fwd_ret_exec_5d"])[
        cols[:5] + ["fwd_ret_exec_5d_copy"] + cols[5:]]                    # label copy among the features
    with_key = d.assign(row_key=np.arange(len(d), dtype=np.float64))[cols[:5] + ["row_key"] + cols[5:]]
    for bad in (renamed, with_label, with_key):
        with pytest.raises(ContractError, match="are not date, symbol, the declared features"):
            training.fit_model(reseal(ds, bad.reset_index(drop=True)), train, RIDGE)
    reordered = reseal(ds, d[swapped])                                     # another dataset, consistent in itself
    t = training.fit_model(ds, train, RIDGE)
    with pytest.raises(ContractError, match="predicts only on its own dataset"):
        training.predict_model(t, reordered, [50])
    own = training.fit_model(reordered, train, RIDGE)
    assert [f for f, _v in own.metadata["features"]] == swapped[2:5]
    assert own.fitted.coef == pytest.approx([t.fitted.coef[i] for i in (1, 0, 2)], rel=1e-12)   # relabelled
    with pytest.raises(ContractError, match="not float"):
        training.fit_model(reseal(ds, d.astype({cols[2]: object})), train, RIDGE)
    with pytest.raises(ContractError, match=r"not in \(date, symbol\) order"):
        training.fit_model(reseal(ds, d.iloc[::-1].reset_index(drop=True)), train, RIDGE)
    with pytest.raises(ContractError, match="share a key"):
        training.fit_model(reseal(ds, pd.concat([d.iloc[:1], d]).reset_index(drop=True)), train, RIDGE)


def test_non_finite_values_in_used_rows_are_refused_and_never_imputed(ds):
    d = ds.data.copy()
    d.loc[5, d.columns[3]] = np.nan
    d.loc[30, "fwd_ret_exec_5d"] = np.inf
    bad = reseal(ds, d)
    with pytest.raises(ContractError, match="NaN or infinite feature"):
        training.fit_model(bad, list(range(10)), RIDGE)
    with pytest.raises(ContractError, match="NaN or infinite label"):
        training.fit_model(bad, list(range(20, 40)), RIDGE)
    t = training.fit_model(bad, list(range(40, 80)), RIDGE)                # unused rows: no effect
    clean = training.fit_model(ds, list(range(40, 80)), RIDGE)
    assert t.fitted == clean.fitted
    p = training.predict_model(t, bad, [30])["prediction"]                # its label is inf: never read
    assert np.isfinite(p).all() and p.iloc[0] == training.predict_model(clean, ds, [30])["prediction"].iloc[0]
    with pytest.raises(ContractError, match="NaN or infinite feature"):
        training.predict_model(t, bad, [4, 5])


def test_the_input_dataset_is_not_mutated(ds):
    before = dataset.content_hash(ds.data), ds.data.copy()
    train, test = _split(ds)
    for spec in SPECS:
        training.predict_model(training.fit_model(ds, train, spec), ds, test)
    assert dataset.content_hash(ds.data) == before[0]
    pd.testing.assert_frame_equal(ds.data, before[1])


def test_a_tampered_trained_model_is_refused(ds):
    from dataclasses import replace
    train, test = _split(ds)
    t = training.fit_model(ds, train, RIDGE)
    fitted = replace(t.fitted, intercept=t.fitted.intercept + 1.0)
    for bad in (replace(t, fitted=fitted), replace(t, spec=ModelSpec("ridge", {"alpha": 0.6})),
                replace(t, model_id="0" * 16)):
        with pytest.raises(ContractError, match="model_id"):
            training.predict_model(bad, ds, test)
    with pytest.raises(ContractError, match="fitted state is ridge"):
        training.predict_model(replace(t, spec=ModelSpec("mean")), ds, test)


def test_the_loaded_dataset_trains_the_same_model_and_seed_does_not_change_the_fit(ds, tmp_path):
    train, _ = _split(ds)
    loaded = dataset.read_dataset(dataset.write_dataset(ds, tmp_path / "ds"))
    t = training.fit_model(ds, train, RIDGE)
    assert training.fit_model(loaded, train, RIDGE).model_id == t.model_id
    seeded = training.fit_model(ds, train, ModelSpec("ridge", {"alpha": 0.5}, seed=7))
    assert seeded.fitted == t.fitted and seeded.model_id != t.model_id


def test_model_id_is_stable_across_processes(ds, tmp_path):
    train, _ = _split(ds)
    path = dataset.write_dataset(ds, tmp_path / "ds")
    ids = {training.fit_model(dataset.read_dataset(path), train, spec).model_id for spec in SPECS}
    code = ("import sys\nfrom quant_research.ml import dataset, training\nfrom quant_research.ml.contracts import "
            "ModelSpec\nd = dataset.read_dataset(sys.argv[1])\nn = len(d[0])\n"
            "for s in (ModelSpec('mean'), ModelSpec('ridge', {'alpha': 0.5}), "
            "ModelSpec('elasticnet', {'alpha': 0.0005, 'l1_wt': 0.5})):\n"
            "    print(training.fit_model(d, list(range(n // 2)), s).model_id)\n")
    out = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True, check=True)
    assert set(out.stdout.split()) == ids and len(ids) == 3

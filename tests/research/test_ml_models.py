"""Baseline estimators and ModelSpec (docs/ml-plan.md, PR #12): exact numerics, unpenalized intercepts, standardization
from the training rows only, refusal of anything non-finite."""

import numpy as np
import pytest
from statsmodels.regression.linear_model import OLS

from quant_research.ml import models
from quant_research.ml.contracts import ContractError, ExperimentSpec, ModelSpec

RIDGE = ModelSpec("ridge", {"alpha": 0.1})
LASSO = ModelSpec("elasticnet", {"alpha": 0.1, "l1_wt": 1})


def _problem(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)) * [1.0, 10.0, 0.01, 1.0] + [0.0, 5.0, -2.0, 1.0]     # poorly scaled, offset
    y = X @ [1.0, 0.1, 50.0, 0.0] + 3.0 + rng.normal(size=n)                          # feature 3 is pure noise
    return X, y


def _standardized(f: models.FittedModel, X):
    return (X - np.array(f.means)) / np.array(f.scales)


def test_model_spec_validates_and_normalizes():
    assert ModelSpec("ridge", {"alpha": 1}).version == ModelSpec("ridge", {"alpha": 1.0}).version
    assert ModelSpec("ridge", {"alpha": 1}, seed=1).version != ModelSpec("ridge", {"alpha": 1}).version
    assert ModelSpec("mean").params == {}
    for bad, why in [(lambda: ModelSpec("lasso", {}), "approved family"),
                     (lambda: ModelSpec("ridge", {}), "exactly the parameters"),
                     (lambda: ModelSpec("ridge", {"alpha": 1, "beta": 2}), "exactly the parameters"),
                     (lambda: ModelSpec("mean", {"alpha": 1}), "exactly the parameters"),
                     (lambda: ModelSpec("ridge", {"alpha": 0}), "alpha must be > 0"),
                     (lambda: ModelSpec("ridge", {"alpha": True}), "alpha must be"),
                     (lambda: ModelSpec("ridge", {"alpha": "1"}), "alpha must be"),
                     (lambda: ModelSpec("ridge", {"alpha": float("inf")}), "finite"),
                     (lambda: ModelSpec("elasticnet", {"alpha": 1, "l1_wt": 0}), r"l1_wt must be in \(0, 1\]"),
                     (lambda: ModelSpec("elasticnet", {"alpha": 1, "l1_wt": 1.5}), r"l1_wt must be in \(0, 1\]"),
                     (lambda: ModelSpec("ridge", {"alpha": 1}, seed=-1), "seed must be >= 0"),
                     (lambda: ModelSpec("ridge", {"alpha": 1}, seed=True), "seed must be int"),
                     (lambda: ModelSpec("ridge", [("alpha", 1)]), "params must be a mapping")]:
        with pytest.raises(ContractError, match=why):
            bad()
    with pytest.raises(TypeError):
        ModelSpec("ridge", {"alpha": 1}).params["alpha"] = 2                          # deeply immutable


def test_mean_is_a_reference_model_not_an_experiment_model():
    import inspect
    assert "mean" not in inspect.getsource(ExperimentSpec) and "mean" not in __import__(
        "quant_research.ml.contracts", fromlist=["MODELS"]).MODELS
    X, y = _problem()
    f = models.fit(ModelSpec("mean"), X, y)
    assert f.intercept == float(y.mean()) and f.coef == (0.0,) * 4
    assert (models.predict(f, X[:5] * 1000) == float(y.mean())).all()


def test_ridge_equals_an_independent_solution_with_an_unpenalized_intercept():
    X, y = _problem()
    f = models.fit(RIDGE, X, y)
    Z = np.column_stack([np.ones(len(y)), _standardized(f, X)])
    sm = OLS(y, Z).fit_regularized(method="elastic_net", alpha=0.1 * np.r_[0, 1, 1, 1, 1], L1_wt=0.0)
    np.testing.assert_allclose(np.r_[f.intercept, f.coef], sm.params, rtol=0, atol=1e-12)
    np.testing.assert_allclose(models.predict(f, X), Z @ sm.params, rtol=0, atol=1e-10)   # standardized inputs
    # augmented least squares: rows sqrt(n * alpha) * I added for the slopes only
    n, k = X.shape
    A = np.vstack([Z, np.column_stack([np.zeros(k), np.sqrt(n * 0.1) * np.eye(k)])])
    b = np.linalg.lstsq(A, np.r_[y, np.zeros(k)], rcond=None)[0]
    np.testing.assert_allclose(np.r_[f.intercept, f.coef], b, rtol=0, atol=1e-10)
    assert f.intercept == pytest.approx(y.mean(), abs=1e-12)                       # never shrunk
    big = models.fit(ModelSpec("ridge", {"alpha": 1e6}), X, y)
    assert max(map(abs, big.coef)) < 1e-4 and big.intercept == pytest.approx(y.mean(), abs=1e-12)


def test_elasticnet_lasso_zeroes_noise_and_keeps_the_intercept_unpenalized():
    X, y = _problem()
    f = models.fit(LASSO, X, y)
    assert f.coef[3] == 0.0 and all(c != 0.0 for c in f.coef[:3])
    assert f.intercept == pytest.approx(y.mean(), abs=1e-8)
    huge = models.fit(ModelSpec("elasticnet", {"alpha": 1e3, "l1_wt": 1}), X, y)
    assert huge.coef == (0.0,) * 4 and huge.intercept == pytest.approx(y.mean(), abs=1e-8)
    mixed = models.fit(ModelSpec("elasticnet", {"alpha": 0.1, "l1_wt": 0.5}), X, y)
    assert models.fit(ModelSpec("elasticnet", {"alpha": 0.1, "l1_wt": 0.5}), X, y) == mixed     # deterministic


def test_elasticnet_that_does_not_converge_is_refused(monkeypatch):
    X, y = _problem()
    monkeypatch.setitem(models.SOLVER, "elasticnet", {"maxiter": 1, "cnvrg_tol": 1e-30})
    with pytest.raises(ContractError, match="did not converge"):
        models.fit(ModelSpec("elasticnet", {"alpha": 0.001, "l1_wt": 0.5}), X, y)


def test_standardization_uses_the_training_rows_and_constants_contribute_nothing():
    X, y = _problem()
    X[:, 2] = 0.1                                                                      # constant in training
    f = models.fit(RIDGE, X, y)
    np.testing.assert_array_equal(f.means, np.where([True, True, False, True], X.mean(0), 0.1))
    np.testing.assert_array_equal(f.scales, np.where([True, True, False, True], X.std(0, ddof=0), 1.0))
    assert f.coef[2] == 0.0
    assert all(c == 0.0 for c in models.fit(LASSO, np.full((10, 2), 3.3), np.arange(10.0)).coef)


def test_collinear_features_stay_finite():
    X, y = _problem()
    X = np.column_stack([X, X[:, 0], 2 * X[:, 0]])
    for spec in (RIDGE, LASSO, ModelSpec("elasticnet", {"alpha": 0.1, "l1_wt": 0.3})):
        f = models.fit(spec, X, y)
        assert np.isfinite(models.predict(f, X)).all()


@pytest.mark.parametrize("X, y, why", [
    (np.ones((3, 0)), np.ones(3), "at least one row and one feature"),
    (np.ones((0, 2)), np.ones(0), "at least one row and one feature"),
    (np.ones(3), np.ones(3), "2-D float64"),
    (np.ones((3, 2), dtype=np.float32), np.ones(3), "2-D float64"),
    (np.array([[1.0, np.nan], [2.0, 1.0]]), np.ones(2), "NaN or infinite"),
    (np.array([[1.0, np.inf], [2.0, 1.0]]), np.ones(2), "NaN or infinite"),
    (np.ones((2, 2)), np.array([1.0, np.nan]), "y has NaN"),
    (np.ones((2, 2)), np.ones(3), "one value per row"),
    (np.ones((2, 2)), np.ones(2, dtype=np.int64), "one value per row"),
])
def test_bad_inputs_are_refused_never_imputed(X, y, why):
    with pytest.raises(ContractError, match=why):
        models.fit(RIDGE, X, y)


def test_predict_checks_shape_and_values_and_fitted_state_is_checked():
    X, y = _problem()
    f = models.fit(RIDGE, X, y)
    with pytest.raises(ContractError, match="X has 3 features"):
        models.predict(f, X[:, :3])
    with pytest.raises(ContractError, match="NaN or infinite"):
        models.predict(f, np.where(np.eye(len(X), 4) == 1, np.nan, X))
    for bad in ({"intercept": 1}, {"intercept": np.float64(1.0)}, {"coef": (1, 0.0, 0.0, 0.0)}):
        with pytest.raises(ContractError, match="finite floats"):
            models.FittedModel(**({"model": "ridge", "means": f.means, "scales": f.scales, "coef": f.coef,
                                   "intercept": f.intercept} | bad))
    j = f.to_json()
    assert models.FittedModel.from_json(j) == f                                      # exact round trip
    for bad in ({**j, "coef": j["coef"][:3]}, {**j, "scales": [(0.0).hex()] * 4}, {**j, "intercept": "inf"},
                {**j, "intercept": 1.5}, {**j, "means": "x"}, {**j, "extra": 1}):
        with pytest.raises(ContractError):
            models.FittedModel.from_json(bad)

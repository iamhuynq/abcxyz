"""Baseline estimators as pure functions over numpy arrays (docs/ml-plan.md, PR #12).

Every model is a linear predictor on standardized features: prediction = intercept + sum_j z_j * coef_j with
z_j = (x_j - mean_j) / scale_j, summed feature by feature in order, so each row's prediction is exact and
independent of the batch it is computed in.
Means and scales come from the training rows only (ddof 0). A feature constant over the training rows gets its
value as mean and scale 1, so centered it is exactly 0 and contributes nothing. Nothing is imputed: any non-finite
input or fitted number is refused.

- mean: coef 0, intercept = mean of the training labels (a sanity reference, not an experiment model).
- ridge: minimizes 0.5 * RSS / n + alpha * |b|_2^2 / 2, solved as (Xs'Xs + n * alpha * I) b = Xs'yc. The
  intercept (the training label mean) is not penalized.
- elasticnet: statsmodels OLS.fit_regularized(method="elastic_net") on [1, Xs], minimizing
  0.5 * RSS / n + alpha * ((1 - l1_wt) * |b|_2^2 / 2 + l1_wt * |b|_1), with a penalty weight of 0 on the intercept
  column. Coordinate descent with fixed solver settings (SOLVER); a fit that does not converge is refused.

The fitted state is a few numbers (FittedModel); it is stored exactly as float.hex strings, never pickled.
"""

import math
import warnings
from dataclasses import dataclass

import numpy as np

from quant_research.ml.contracts import (
    MODELS,
    REFERENCE_MODELS,
    ContractError,
    ModelSpec,
)

SOLVER = {"elasticnet": {"maxiter": 1000, "cnvrg_tol": 1e-8}}          # coordinate descent (statsmodels default)


@dataclass(frozen=True)
class FittedModel:
    model: str
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coef: tuple[float, ...]
    intercept: float

    def __post_init__(self) -> None:
        for name in ("means", "scales", "coef"):      # own immutable copies: the caller's lists are never kept
            v = getattr(self, name)
            if not isinstance(v, (list, tuple)):
                raise ContractError(f"fitted {name} must be a list or tuple of floats, not {type(v).__name__}")
            object.__setattr__(self, name, tuple(v))
        if self.model not in MODELS + REFERENCE_MODELS:
            raise ContractError(f"fitted model {self.model!r} is not an approved family")
        k = len(self.coef)
        if k == 0 or not (len(self.means) == len(self.scales) == k):
            raise ContractError(f"fitted state has inconsistent shapes {len(self.means)}, {len(self.scales)}, {k}")
        values = (*self.means, *self.scales, *self.coef, self.intercept)
        if not all(type(v) is float and math.isfinite(v) for v in values):
            raise ContractError("fitted state must be finite floats")
        if not all(s > 0 for s in self.scales):
            raise ContractError("fitted scales must be > 0")

    def to_json(self) -> dict:
        """Exact: every number as float.hex."""
        return {"model": self.model, "means": [v.hex() for v in self.means], "scales": [v.hex() for v in self.scales],
                "coef": [v.hex() for v in self.coef], "intercept": self.intercept.hex()}

    @classmethod
    def from_json(cls, d) -> "FittedModel":
        if not isinstance(d, dict) or set(d) != {"model", "means", "scales", "coef", "intercept"}:
            raise ContractError("fitted state must have exactly model, means, scales, coef, intercept")

        def num(v) -> float:
            if not isinstance(v, str):
                raise ContractError(f"fitted number {v!r} is not a float.hex string")
            try:
                return float.fromhex(v)
            except ValueError:
                raise ContractError(f"fitted number {v!r} is not a float.hex string") from None

        def vec(v) -> tuple[float, ...]:
            if not isinstance(v, list):
                raise ContractError("fitted vectors must be lists")
            return tuple(num(x) for x in v)
        if not isinstance(d["model"], str):
            raise ContractError("fitted model name must be a string")
        return cls(d["model"], vec(d["means"]), vec(d["scales"]), vec(d["coef"]), num(d["intercept"]))


def _matrix(X, k: int | None = None) -> np.ndarray:
    if not isinstance(X, np.ndarray) or X.ndim != 2 or X.dtype != np.float64:
        raise ContractError("X must be a 2-D float64 array")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ContractError(f"X has shape {X.shape}: at least one row and one feature are needed")
    if k is not None and X.shape[1] != k:
        raise ContractError(f"X has {X.shape[1]} features, the model {k}")
    if not np.isfinite(X).all():
        raise ContractError("X has NaN or infinite values; nothing is imputed")
    return np.ascontiguousarray(X)                    # one memory layout: results do not depend on the caller's


def _standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    constant = (X == X[0]).all(axis=0)
    means = np.where(constant, X[0], X.mean(axis=0))
    scales = np.where(constant, 1.0, X.std(axis=0, ddof=0))
    if not (np.isfinite(means).all() and np.isfinite(scales).all() and (scales > 0).all()):
        raise ContractError("feature standardization is not finite")
    return means, scales


def _elasticnet(Xs: np.ndarray, y: np.ndarray, alpha: float, l1_wt: float) -> tuple[float, np.ndarray]:
    from statsmodels.regression.linear_model import OLS
    Z = np.column_stack([np.ones(len(y)), Xs])
    weights = alpha * np.r_[0.0, np.ones(Xs.shape[1])]               # the intercept is not penalized
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = OLS(y, Z).fit_regularized(method="elastic_net", alpha=weights, L1_wt=l1_wt, **SOLVER["elasticnet"])
    if not bool(getattr(r, "converged", False)):
        raise ContractError(f"elasticnet did not converge within {SOLVER['elasticnet']['maxiter']} iterations")
    params = np.asarray(r.params, dtype=np.float64)
    return float(params[0]), params[1:]


def fit(spec: ModelSpec, X: np.ndarray, y: np.ndarray) -> FittedModel:
    if not isinstance(spec, ModelSpec):
        raise ContractError("spec must be a ModelSpec")
    X = _matrix(X)
    if not isinstance(y, np.ndarray) or y.dtype != np.float64 or y.shape != (X.shape[0],):
        raise ContractError("y must be a 1-D float64 array with one value per row of X")
    if not np.isfinite(y).all():
        raise ContractError("y has NaN or infinite values; nothing is imputed")
    means, scales = _standardize(X)
    Xs = (X - means) / scales
    n, k = X.shape
    ybar = float(y.mean())
    try:
        if spec.model == "mean":
            intercept, coef = ybar, np.zeros(k)
        elif spec.model == "ridge":
            a = Xs.T @ Xs + n * spec.params["alpha"] * np.eye(k)
            intercept, coef = ybar, np.linalg.solve(a, Xs.T @ (y - ybar))
        elif spec.model == "elasticnet":
            intercept, coef = _elasticnet(Xs, y, spec.params["alpha"], spec.params["l1_wt"])
        else:                                                             # unreachable: ModelSpec validates
            raise ContractError(f"no estimator for {spec.model!r}")
    except np.linalg.LinAlgError as e:
        raise ContractError(f"{spec.model} fit failed: {e}") from None
    return FittedModel(spec.model, tuple(map(float, means)), tuple(map(float, scales)), tuple(map(float, coef)),
                       float(intercept))


def predict(fitted: FittedModel, X: np.ndarray) -> np.ndarray:
    X = _matrix(X, len(fitted.coef))
    out = np.full(X.shape[0], fitted.intercept)
    for j, (m, s, c) in enumerate(zip(fitted.means, fitted.scales, fitted.coef, strict=True)):
        out = out + ((X[:, j] - m) / s) * c         # elementwise, feature by feature: a row's prediction does not
    if not np.isfinite(out).all():                  # depend on the other rows of the batch (no BLAS blocking)
        raise ContractError("predictions are not finite")
    return out

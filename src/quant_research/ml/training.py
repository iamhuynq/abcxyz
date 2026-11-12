"""Fit and predict on explicit row positions of one verified dataset (docs/ml-plan.md, PR #12).

No split, no fold, no purge here: the caller (PR #13) chooses the rows. This module checks them.

- The dataset is verified on every call (dataset.verify_dataset: content hash and fingerprint), for a
  build_dataset result and a read_dataset artifact alike. Its rows must be in (date, symbol) order with unique keys
  and its columns exactly date, symbol, the ordered features, the label, label_start, label_end.
- Indices are row positions in the dataset: a non-empty 1-D sequence of integers (no bool, no float), no
  duplicates, within range. Training indices must be strictly increasing (never sorted silently); prediction
  indices may be in any order and the output keeps it.
- The estimator sees only the feature columns of the selected rows (and, for fitting, their labels). Labels are
  attached to predictions afterwards, for evaluation only. Nothing is filled or coerced.
- A trained model predicts only on the dataset it was trained on (same verified fingerprint, same ordered
  features and source versions, same label).
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from quant_research import provenance
from quant_research.ml import models
from quant_research.ml.canonical import digest
from quant_research.ml.contracts import ContractError, ModelSpec
from quant_research.ml.dataset import DatasetResult, verify_dataset
from quant_research.ml.keys import check_unique

CODE_FILES = ("canonical.py", "contracts.py", "dataset.py", "models.py", "training.py")


@dataclass(frozen=True)
class Dataset:
    """A verified dataset view: what fit and predict read."""
    data: pd.DataFrame
    metadata: Mapping
    fingerprint: str
    features: tuple[str, ...]
    label: str


@dataclass(frozen=True)
class TrainedModel:
    spec: ModelSpec
    fitted: models.FittedModel
    metadata: Mapping                # JSON-native: dataset identity, features, label, training summary, environment
    model_id: str


def code_hash() -> str:
    h = hashlib.sha256()
    for name in CODE_FILES:
        h.update(name.encode() + b"\0" + (Path(__file__).parent / name).read_bytes() + b"\0")
    return h.hexdigest()[:16]


def model_identity(spec: ModelSpec, fitted: models.FittedModel, metadata: Mapping) -> str:
    """Canonical digest of the spec, the exact fitted numbers and the metadata (no timestamp, no path)."""
    return digest({"spec": spec, "fitted": fitted.to_json(), "metadata": metadata})


def check_trained(trained: TrainedModel) -> None:
    if not isinstance(trained, TrainedModel):
        raise ContractError("expected a TrainedModel")
    if trained.fitted.model != trained.spec.model:
        raise ContractError(f"fitted state is {trained.fitted.model}, the spec {trained.spec.model}")
    if len(trained.fitted.coef) != len(trained.metadata["features"]):
        raise ContractError("fitted state does not match the number of features")
    if model_identity(trained.spec, trained.fitted, trained.metadata) != trained.model_id:
        raise ContractError("the trained model does not match its model_id")


def open_dataset(ds) -> Dataset:
    """A build_dataset result, or the (data, document) pair of read_dataset, verified."""
    if isinstance(ds, DatasetResult):
        data, meta, fp = ds.data, ds.metadata, ds.fingerprint
    elif isinstance(ds, tuple) and len(ds) == 2 and isinstance(ds[0], pd.DataFrame) and isinstance(ds[1], Mapping):
        data, meta, fp = ds[0], ds[1].get("metadata"), ds[1].get("fingerprint")
    else:
        raise ContractError("dataset must be a DatasetResult or the (data, document) pair of read_dataset")
    if not isinstance(meta, Mapping):
        raise ContractError("dataset has no metadata")
    verify_dataset(data, meta, fp)
    cols, label, versions = list(meta.get("columns", [])), meta.get("label"), meta.get("feature_versions", {})
    features = tuple(cols[2:-3])
    if (not features or cols != ["date", "symbol", *features, label, "label_start", "label_end"]
            or list(data.columns) != cols or set(features) != set(versions) or len(set(cols)) != len(cols)):
        raise ContractError(f"dataset columns {list(data.columns)} are not date, symbol, the declared features, "
                            f"the label {label!r}, label_start, label_end")
    for c in (*features, label):
        if data[c].dtype.kind != "f":
            raise ContractError(f"column {c} has dtype {data[c].dtype}, not float: nothing is coerced")
    check_unique(data)
    keys = pd.MultiIndex.from_arrays([pd.to_datetime(data["date"]), data["symbol"]])
    if not keys.is_monotonic_increasing:
        raise ContractError("dataset rows are not in (date, symbol) order; rows are never sorted silently")
    return Dataset(data, meta, fp, features, label)


def _indices(idx, n: int, increasing: bool, what: str) -> np.ndarray:
    if isinstance(idx, (str, bytes, Mapping)) or not hasattr(idx, "__len__"):
        raise ContractError(f"{what} must be a sequence of row positions")
    if not isinstance(idx, np.ndarray) and not all(
            isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in idx):
        raise ContractError(f"{what} must be integers (no bool, no float)")
    a = np.asarray(idx)
    if a.ndim != 1 or a.size == 0 or (a.dtype.kind not in "iu"):
        raise ContractError(f"{what} must be a non-empty 1-D integer sequence, not {a.dtype} with shape {a.shape}")
    if a.min() < 0 or a.max() >= n:
        raise ContractError(f"{what} must be within 0..{n - 1}")
    if len(np.unique(a)) != len(a):
        raise ContractError(f"{what} has duplicates")
    if increasing and not (np.diff(a) > 0).all():
        raise ContractError(f"{what} must be strictly increasing; they are never sorted silently")
    return a.astype(np.int64)


def _features(ds: Dataset, rows: np.ndarray) -> np.ndarray:
    X = ds.data[list(ds.features)].to_numpy(dtype=np.float64)[rows]      # feature columns only
    if not np.isfinite(X).all():
        raise ContractError("a selected row has a NaN or infinite feature; nothing is imputed")
    return X


def _keys_hash(ds: Dataset, rows: np.ndarray) -> str:
    sel = ds.data.iloc[rows]
    return digest([[d, s] for d, s in zip(pd.to_datetime(sel["date"]).dt.date, sel["symbol"], strict=True)])


def _day(v) -> str:
    return pd.Timestamp(v).date().isoformat()


def fit_model(dataset, train_indices, spec: ModelSpec) -> TrainedModel:
    if not isinstance(spec, ModelSpec):
        raise ContractError("spec must be a ModelSpec")
    ds = open_dataset(dataset)
    rows = _indices(train_indices, len(ds.data), increasing=True, what="train_indices")
    X = _features(ds, rows)
    y = ds.data[ds.label].to_numpy(dtype=np.float64)[rows]
    if not np.isfinite(y).all():
        raise ContractError("a training row has a NaN or infinite label; nothing is imputed")
    fitted = models.fit(spec, X, y)
    sel = ds.data.iloc[rows]
    meta = ds.metadata
    metadata = json.loads(json.dumps({
        "label": ds.label, "horizon": meta.get("horizon"),
        "features": [[f, meta["feature_versions"][f]] for f in ds.features],
        "dataset_fingerprint": ds.fingerprint,
        "dataset_provenance": dict(meta),
        "training": {"rows": len(rows), "keys_hash": _keys_hash(ds, rows),
                     "first_date": _day(min(sel["date"])), "last_date": _day(max(sel["date"])),
                     "first_label_start": _day(min(sel["label_start"])),
                     "last_label_end": _day(max(sel["label_end"]))},
        "solver": models.SOLVER.get(spec.model, {}),
        "environment": provenance.environment(), "code_hash": code_hash()}, default=str))
    metadata = MappingProxyType(metadata)
    return TrainedModel(spec, fitted, metadata, model_identity(spec, fitted, metadata))


def predict_model(trained: TrainedModel, dataset, prediction_indices) -> pd.DataFrame:
    """Predictions in the order of `prediction_indices`: row, date, symbol, label_name, prediction, then the label
    value (attached after prediction, for evaluation only), model_id, dataset_fingerprint."""
    check_trained(trained)
    ds = open_dataset(dataset)
    m = trained.metadata
    if ds.fingerprint != m["dataset_fingerprint"]:
        raise ContractError(f"the model was trained on dataset {m['dataset_fingerprint']}, not {ds.fingerprint}: "
                            "it predicts only on its own dataset")
    if ([[f, ds.metadata["feature_versions"][f]] for f in ds.features] != list(map(list, m["features"]))
            or ds.label != m["label"]):
        raise ContractError("the dataset's features or label differ from the model's")
    rows = _indices(prediction_indices, len(ds.data), increasing=False, what="prediction_indices")
    pred = models.predict(trained.fitted, _features(ds, rows))
    sel = ds.data.iloc[rows]
    out = pd.DataFrame({"row": rows, "date": sel["date"].to_numpy(), "symbol": sel["symbol"].to_numpy(),
                        "label_name": ds.label, "prediction": pred})
    out["label"] = sel[ds.label].to_numpy()                             # attached after prediction
    out["model_id"] = trained.model_id
    out["dataset_fingerprint"] = ds.fingerprint
    return out

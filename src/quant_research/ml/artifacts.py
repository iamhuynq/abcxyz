"""Model artifacts: one directory holding model.json, canonical JSON only (docs/ml-plan.md, PR #12).

Trust boundary: loading parses JSON and numbers; no code is ever loaded or executed (no pickle, no joblib). The
fitted numbers are stored as float.hex, so a loaded model predicts exactly like the saved one. `model_id` is
recomputed on load from the spec, the fitted numbers and the metadata, and any edit is refused. It is an
integrity check, not a signature: someone who rewrites the whole file can recompute it. `created_at` is for audit
only and is not part of the identity.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

from quant_research.ml import publish
from quant_research.ml.contracts import ContractError, ModelSpec
from quant_research.ml.models import FittedModel
from quant_research.ml.training import TrainedModel, check_trained

SCHEMA_VERSION = "model/1"
FIELDS = {"artifact_schema_version", "model_id", "created_at", "spec", "fitted", "metadata"}


def save_model(trained: TrainedModel, path: Path) -> Path:
    """Publish atomically at `path`, which must not exist; verified by loading it back before it is visible."""
    check_trained(trained)
    doc = {"artifact_schema_version": SCHEMA_VERSION, "model_id": trained.model_id,
           "created_at": datetime.now(UTC).isoformat(),
           "spec": {"model": trained.spec.model, "params": dict(trained.spec.params), "seed": trained.spec.seed},
           "fitted": trained.fitted.to_json(), "metadata": dict(trained.metadata)}
    text = json.dumps(doc, sort_keys=True, indent=1, allow_nan=False)

    def verify(tmp: Path) -> None:
        if load_model(tmp).model_id != trained.model_id:
            raise ContractError("the written artifact does not reload to the same model")
    path = Path(path)
    with publish.locked(path):
        return publish.publish_dir(path, lambda tmp: (tmp / "model.json").write_text(text), verify,
                                   "a model artifact")


def load_model(path: Path) -> TrainedModel:
    path = Path(path)
    try:
        doc = json.loads((path / "model.json").read_text())
    except (OSError, ValueError) as e:
        raise ContractError(f"{path}: unreadable model artifact ({type(e).__name__})") from None
    if not isinstance(doc, dict) or set(doc) != FIELDS:
        raise ContractError(f"{path}: a model artifact has exactly the fields {sorted(FIELDS)}")
    if doc["artifact_schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"{path}: unsupported artifact schema {doc['artifact_schema_version']!r}")
    s, meta = doc["spec"], doc["metadata"]
    if not isinstance(s, dict) or set(s) != {"model", "params", "seed"} or not isinstance(meta, dict):
        raise ContractError(f"{path}: malformed spec or metadata")
    spec = ModelSpec(s["model"], s["params"], s["seed"])                # validated again
    fitted = FittedModel.from_json(doc["fitted"])                        # shapes, finite floats
    try:
        trained = TrainedModel(spec, fitted, MappingProxyType(meta), doc["model_id"])   # structure and metadata
        check_trained(trained)                                                         # and model_id
    except ContractError as e:
        raise ContractError(f"{path}: {e}") from None
    return trained

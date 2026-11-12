"""Deterministic, versioned hashes for ML contracts (docs/ml-plan.md section 6.3).

Canonical JSON (sorted keys, fixed separators), with dates and datetimes as ISO strings, tuples as lists and
dataclasses as their fields; floats must be finite. Any other type is refused rather than stringified. Every
payload is wrapped with the contract version, so a contract change changes every hash. The digest is SHA-256
(16 hex characters, like provenance.config_hash). Python's hash() is never used: it is salted per process.
"""

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from datetime import date, datetime

CONTRACT = "ml/1"


class HashError(TypeError):
    pass


def normalize(x):
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if isinstance(x, float):
        if not math.isfinite(x):
            raise HashError(f"non-finite float {x!r} cannot be hashed")
        return x
    if isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, date):
        return x.isoformat()
    if isinstance(x, (list, tuple)):
        return [normalize(v) for v in x]
    if isinstance(x, Mapping):                   # dict, or the read-only mapping of a frozen spec
        if not all(isinstance(k, str) for k in x):
            raise HashError("dictionary keys must be strings")
        return {k: normalize(v) for k, v in x.items()}
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: normalize(getattr(x, f.name)) for f in dataclasses.fields(x)}
    raise HashError(f"type {type(x).__name__} has no canonical form")


def canonical(payload, contract: str = CONTRACT) -> str:
    return json.dumps({"contract": contract, "payload": normalize(payload)}, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def digest(payload, contract: str = CONTRACT) -> str:
    return hashlib.sha256(canonical(payload, contract).encode()).hexdigest()[:16]

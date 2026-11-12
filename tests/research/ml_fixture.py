"""A small synthetic research dataset for the ML model tests (PR #12), built with the PR #11 builder."""

from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType

import duckdb
import pandas as pd

from quant_research.build import build
from quant_research.ml import dataset
from quant_research.ml.contracts import DatasetSpec, FeatureSpec, LabelSpec
from tests.research.synthetic import build_synthetic_warehouse

F = FeatureSpec.current
FEATURES = (F("stock_features", "return_5d"), F("stock_features", "volume_ratio_20"),
            F("stock_features", "volatility_20"))
SPEC = DatasetSpec(FEATURES, LabelSpec("fwd_ret_exec_5d", 5), date(2022, 1, 3), date(2023, 12, 29))


def build_dataset(d: Path) -> dataset.DatasetResult:
    build_synthetic_warehouse(d / "w.duckdb", start=date(2022, 1, 3))
    build(d / "w.duckdb", d / "r.duckdb", datetime(2026, 11, 12, 9, 0, tzinfo=UTC))
    with duckdb.connect(str(d / "r.duckdb"), read_only=True) as con:
        build_id = con.execute("SELECT build_id FROM feature_builds ORDER BY built_at DESC LIMIT 1").fetchone()[0]
        return dataset.build_dataset(con, SPEC, build_id)


def reseal(r: dataset.DatasetResult, data: pd.DataFrame, **meta_changes) -> dataset.DatasetResult:
    """A self-consistent dataset (content hash and fingerprint recomputed) around changed data: what someone who
    rewrites every part would produce. Used to test the checks that a fingerprint alone cannot catch."""
    meta = dict(r.metadata) | {"columns": list(data.columns)} | meta_changes
    meta["content_hash"] = dataset.content_hash(data)
    return replace(r, data=data, metadata=MappingProxyType(meta),
                   fingerprint=dataset.fingerprint(meta, meta["content_hash"]))

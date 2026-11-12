"""ML dataset builder (docs/ml-plan.md, PR #11). Reads research.duckdb read-only; writes nothing to any database.

One candidate row per daily_panel row (date, symbol) in the dataset period. Each candidate gets exactly one status:
kept, or the first exclusion reason that applies, in the declared order of REASONS. Nothing is filled, clipped or
silently dropped. Label intervals come from each symbol's full daily_panel series (before any filter), with the same
window as 04_stock_targets.sql: label_start = lead(date, 1), label_end = lead(date, h), so
adj_close(label_end) / adj_open(label_start) - 1 reproduces fwd_ret_exec_{h}d.
"""

import fcntl
import hashlib
import json
import math
import os
import shutil
import socket
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType

import duckdb
import numpy as np
import pandas as pd

from quant_research.ml import allowlist
from quant_research.ml.builds import build_identity, build_source_versions, check_build
from quant_research.ml.canonical import CONTRACT, digest
from quant_research.ml.contracts import RESEARCH_END, UNIVERSES, ContractError, DatasetSpec
from quant_research.ml.keys import check_point_in_time, check_unique

SCHEMA_VERSION = "dataset/1"
REASONS = ("outside_universe", "no_label_interval", "entry_blocked", "price_jump_in_window", "label_crosses_period",
           "label_after_dataset_end", "label_null", "feature_null")


@dataclass(frozen=True)
class DatasetResult:
    data: pd.DataFrame                                   # kept rows: date, symbol, features, label, interval
    metadata: MappingProxyType
    fingerprint: str
    exclusions: tuple[tuple[str, int, float], ...]       # (reason, count, share of input rows), in REASONS order
    input_rows: int
    kept_rows: int
    excluded_rows: int


def _feature_sql(spec: DatasetSpec, s: str) -> tuple[list[str], list[str], list]:
    """(select expressions, join clauses, bind values). Identifiers come only from validated FeatureSpecs."""
    cols, joins, binds = [], [], []
    for k, f in enumerate(spec.features):
        alias = f'"{f.name}"'
        if f.table == "stock_features":
            cols.append(f"sf.{f.column} AS {alias}")
        elif f.table == "stock_factors":
            factor, field = f.column.split(".")
            joins.append(f"LEFT JOIN (SELECT date, symbol, {field} AS v FROM {s}stock_factors WHERE factor = ?) "
                         f"fx{k} ON fx{k}.date = c.date AND fx{k}.symbol = c.symbol")
            binds.append(factor)
            cols.append(f"fx{k}.v AS {alias}")
        else:                                            # market_regimes: one row per date
            cols.append(f"mr.{f.column} AS {alias}")
    return cols, joins, binds


def _load(con: duckdb.DuckDBPyConnection, spec: DatasetSpec, s: str) -> pd.DataFrame:
    h, label = spec.label.horizon, spec.label.column
    cols, joins, binds = _feature_sql(spec, s)
    universe = UNIVERSES[spec.universe]
    sql = f"""
        WITH iv AS (SELECT symbol, date, period AS decision_period, lead(date, 1) OVER w AS label_start,
                           lead(date, {h}) OVER w AS label_end, lead(period, {h}) OVER w AS label_end_period
                    FROM {s}daily_panel WINDOW w AS (PARTITION BY symbol ORDER BY date)),
        c AS (SELECT symbol, date FROM {s}daily_panel WHERE date BETWEEN ? AND ?)
        SELECT c.date, c.symbol, sf.date AS feature_date, coalesce(sf.in_universe, false) AS in_universe,
               iv.label_start, iv.label_end, t.{label} AS label, t.entry_blocked,
               t.fwd_has_price_jump_{h}d AS price_jump_in_window, iv.decision_period, iv.label_end_period,
               {", ".join(cols)}
        FROM c
        LEFT JOIN iv ON iv.symbol = c.symbol AND iv.date = c.date
        LEFT JOIN (SELECT *, coalesce(({universe}), false) AS in_universe FROM {s}stock_features) sf
               ON sf.symbol = c.symbol AND sf.date = c.date
        LEFT JOIN {s}stock_targets t ON t.symbol = c.symbol AND t.date = c.date
        LEFT JOIN {s}market_regimes mr ON mr.date = c.date
        {" ".join(joins)}
        ORDER BY c.date, c.symbol"""
    return con.execute(sql, [spec.period_start, spec.period_end, *binds]).df()


def _status(df: pd.DataFrame, spec: DatasetSpec) -> pd.Series:
    """Exactly one status per row: the first reason of REASONS that applies, else kept."""
    features = [f.name for f in spec.features]
    end = pd.to_datetime(df["label_end"]).dt.date
    checks = {"outside_universe": ~df["in_universe"].astype(bool),
              "no_label_interval": df["label_end"].isna(),
              "entry_blocked": df["entry_blocked"].eq(True),
              "price_jump_in_window": df["price_jump_in_window"].eq(True),
              # the label ends in another period than its decision (daily_panel's own period rule), or past the
              # research cutoff: checked here from the interval, not taken from stock_targets
              "label_crosses_period": end.notna() & ((df["label_end_period"] != df["decision_period"])
                                                     | (end > RESEARCH_END)),
              "label_after_dataset_end": end.notna() & (end > spec.period_end),
              "label_null": df["label"].isna(),
              "feature_null": df[features].isna().any(axis=1)}
    assert tuple(checks) == REASONS
    status = pd.Series("kept", index=df.index, dtype=object)
    for reason in REASONS:                              # explicit order: first match wins
        status = status.mask((status == "kept") & checks[reason].fillna(False).astype(bool), reason)
    return status


def _cell(v):
    """Canonical value: missing (None, NaN, NaT) as null; bool as bool; integers as int; floats as float.hex with
    -0.0 as 0.0 and +/-inf kept ('inf', '-inf'); dates as ISO dates; datetimes with a time as full ISO."""
    if v is None or v is pd.NaT:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isnan(f):
            return None
        return (0.0 if f == 0.0 else f).hex()
    if isinstance(v, (pd.Timestamp, datetime, np.datetime64)):
        t = pd.Timestamp(v)
        if pd.isna(t):                                   # np.datetime64("NaT") is missing, like pd.NaT
            return None
        return t.date().isoformat() if t == t.normalize() and t.tzinfo is None else t.isoformat()
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, str):
        return v
    raise ContractError(f"value of type {type(v).__name__} has no canonical form")


def content_hash(data: pd.DataFrame) -> str:
    """SHA-256 of the rows sorted by (date, symbol), each serialized canonically in column order (ISO dates,
    floats as float.hex, NULL as null)."""
    h = hashlib.sha256(json.dumps(list(data.columns)).encode())
    rows = data.sort_values(["date", "symbol"], kind="mergesort")
    for row in rows.itertuples(index=False):
        h.update(json.dumps([_cell(v) for v in row], separators=(",", ":")).encode() + b"\n")
    return h.hexdigest()


def fingerprint(provenance: dict, content: str) -> str:
    """The dataset fingerprint: a canonical SHA-256 over the provenance (metadata without content_hash) and the
    logical content hash. Used by both build_dataset and read_dataset. This is an integrity check (any change to the
    data, the content hash, the metadata or the fingerprint is detected), not a digital signature: anyone who can
    rewrite all of them can recompute it."""
    return digest({"provenance": {k: v for k, v in provenance.items() if k != "content_hash"}, "content": content})


def build_dataset(con: duckdb.DuckDBPyConnection, spec: DatasetSpec, build_id: str,
                  schema: str = "") -> DatasetResult:
    """`con`: a read-only connection to research.duckdb (schema "") or a store with it attached as rs ("rs.")."""
    check_build(con, build_id, spec, schema=schema)       # before any dataset read
    ident = build_identity(con, build_id, schema)
    versions = build_source_versions(con, build_id, schema)
    df = _load(con, spec, schema)
    check_unique(df)
    dated = df["feature_date"].notna()               # a candidate without a stock_features row is counted, not dated
    check_point_in_time(df[dated])
    status = _status(df, spec)
    features = [f.name for f in spec.features]
    data = df.loc[status == "kept", ["date", "symbol", *features, "label", "label_start", "label_end"]]
    data = data.rename(columns={"label": spec.label.column}).reset_index(drop=True)
    for col in ("date", "label_start", "label_end"):
        data[col] = pd.to_datetime(data[col]).dt.date
    n = len(df)
    counts = status.value_counts()
    exclusions = tuple((r, int(counts.get(r, 0)), (int(counts.get(r, 0)) / n) if n else 0.0) for r in REASONS)
    excluded = sum(c for _r, c, _s in exclusions)
    if n != len(data) + excluded:
        raise ContractError(f"exclusion accounting does not reconcile: {n} != {len(data)} + {excluded}")
    meta = {"contract": CONTRACT, "schema_version": SCHEMA_VERSION, "spec_fingerprint": spec.fingerprint,
            "columns": list(data.columns), "feature_versions": {f.name: f.version for f in spec.features},
            "label": spec.label.column, "horizon": spec.label.horizon, "build_id": build_id,
            "source_versions": versions, **ident, "period": [spec.period_start, spec.period_end],
            "universe_sql": UNIVERSES[spec.universe], "input_rows": n, "kept_rows": len(data),
            "excluded_rows": excluded, "exclusions": [list(e) for e in exclusions],
            "limitations": list(allowlist.LIMITATIONS), "rows_without_stock_features": int((~dated).sum()),
            "content_hash": content_hash(data)}
    if (data["label_end"] > RESEARCH_END).any():      # invariant, also guaranteed by label_crosses_period
        raise ContractError("a kept label ends after the research cutoff")
    return DatasetResult(data, MappingProxyType(meta), fingerprint(meta, meta["content_hash"]), exclusions, n,
                         len(data), excluded)


def write_dataset(result: DatasetResult, path: Path) -> Path:
    """data.parquet + metadata.json in one directory, renamed into place in one step. `path` must not exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(f".{path.name}.lock")
    fd = _acquire(lock, path)
    try:
        return _publish(result, path)
    finally:
        os.close(fd)                                  # releases the flock; the lock file stays


def _acquire(lock: Path, path: Path) -> int:
    """One publisher per path, with an OS advisory lock (fcntl.flock, exclusive, non-blocking) on a lock file that
    is never deleted. The kernel releases the lock when its holder exits, crashes or is killed, so there is no stale
    lock to recover and no PID bookkeeping. The file is never unlinked: deleting a flock file while another process
    holds it open would let two publishers lock different inodes. Returns the open descriptor; closing it unlocks.
    POSIX only (macOS, Linux)."""
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise FileExistsError(f"{path} is being published by another process (lock {lock})") from None
    try:                                              # diagnostics only: who holds it; nothing reads it
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "since": time.time()}).encode())
    except BaseException:
        os.close(fd)
        raise
    return fd


def _publish(result: DatasetResult, path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"{path} exists: a dataset artifact is never overwritten")
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    tmp.mkdir()
    try:
        result.data.to_parquet(tmp / "data.parquet", index=False)
        (tmp / "metadata.json").write_text(json.dumps({"fingerprint": result.fingerprint,
                                                       "metadata": json.loads(json.dumps(
                                                           dict(result.metadata), default=str))},
                                                      sort_keys=True, indent=1))
        read_dataset(tmp)                                 # verify both files before they become visible
        tmp.rename(path)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return path


def read_dataset(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read an artifact and refuse it unless the data's content hash matches its metadata."""
    path = Path(path)
    doc = json.loads((path / "metadata.json").read_text())
    data = pd.read_parquet(path / "data.parquet")
    meta = doc["metadata"]
    actual = content_hash(data)                       # recomputed from the data, never trusted from the metadata
    if actual != meta.get("content_hash"):
        raise ContractError(f"{path}: the data does not match its metadata (content hash)")
    if doc.get("fingerprint") != fingerprint(meta, actual):
        raise ContractError(f"{path}: the metadata does not match its fingerprint")
    return data, doc

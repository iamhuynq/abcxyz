"""Build research.duckdb from the warehouse: panel -> market -> features -> targets -> views, then factors,
regimes and trading-cost inputs.

The warehouse is ATTACHed READ_ONLY and its file hash is checked before and after the build, so
Phase 2 can never change Phase 1 data. Every build is a full rebuild (adj_ratio is rewritten
retroactively by the source, so incremental feature updates would be wrong).
"""

import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from importlib import resources
from pathlib import Path

import duckdb

from fireant_crawler.store.migrations import plan_schema_changes
from quant_research.backtest.costs import build_cost_inputs
from quant_research.factors import FactorParams, build_factors
from quant_research.factors import version as factors_version
from quant_research.regimes import RegimeParams, build_regimes
from quant_research.regimes import version as regimes_version

FEATURE_SET_VERSION = "v2"  # v2: 2026 split into holdout (<= 2026-10-02) and forward
SQL_PACKAGE = "quant_research"


class BuildError(RuntimeError):
    pass


@dataclass(frozen=True)
class BuildResult:
    build_id: str
    data_as_of: str
    code_hash: str
    rows: dict[str, int]
    seconds: float


def sql_files() -> list[tuple[str, str]]:
    folder = resources.files(SQL_PACKAGE).joinpath("sql")
    files = sorted((p.name, p.read_text(encoding="utf-8")) for p in folder.iterdir() if p.name.endswith(".sql"))
    if not files:
        raise BuildError("No SQL files found in quant_research/sql")
    return files


def feature_build_files() -> list[str]:
    """The code that shapes research.duckdb (its scope is the feature build only, not the research runs)."""
    return [f"sql/{name}" for name, _ in sql_files()] + ["build.py", "factors.py", "regimes.py", "backtest/costs.py"]


def feature_build_hash() -> str:
    """Hash of the feature-build code (stored as feature_builds.code_hash). Research runs store their own,
    wider research_code_hash (quant_research.results), which covers every module of the package."""
    digest = hashlib.sha256()
    for name, text in sql_files():
        digest.update(name.encode())
        digest.update(text.encode())
    for rel in feature_build_files()[len(sql_files()):]:
        digest.update(resources.files(SQL_PACKAGE).joinpath(*rel.split("/")).read_bytes())
    return digest.hexdigest()[:16]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_warehouse(warehouse_path: Path) -> None:
    if not warehouse_path.exists():
        raise BuildError(f"No warehouse at {warehouse_path}; run `fireant backfill` first")
    con = duckdb.connect(str(warehouse_path), read_only=True)
    try:
        changes = plan_schema_changes(con)
    finally:
        con.close()
    if changes:
        raise BuildError("Warehouse schema is out of date; run `fireant migrate --dry-run` then `fireant migrate`")


_BUILDS_DDL = """
CREATE TABLE IF NOT EXISTS feature_builds (
    build_id            VARCHAR PRIMARY KEY,
    feature_set_version VARCHAR NOT NULL,
    built_at            TIMESTAMPTZ NOT NULL,
    data_as_of          DATE,
    warehouse_fetched_at TIMESTAMPTZ,
    code_hash           VARCHAR NOT NULL,
    panel_rows          BIGINT,
    feature_rows        BIGINT,
    target_rows         BIGINT,
    seconds             DOUBLE,
    warehouse_sha       VARCHAR,
    -- Added 2026-11-12 (ML foundation): the effective factor / regime configuration of this build (canonical JSON
    -- of the parameters actually passed) and the definition versions. NULL on older rows: unknown, never default.
    factor_params       VARCHAR,
    regime_params       VARCHAR,
    factors_version     VARCHAR,
    regimes_version     VARCHAR
)"""


def params_json(params) -> str:
    """Canonical JSON of a parameter dataclass (sorted keys, fixed separators)."""
    return json.dumps(dataclasses.asdict(params), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _remove_db(path: Path) -> None:
    for p in (path, path.with_name(path.name + ".wal")):
        p.unlink(missing_ok=True)


def build(warehouse_path: Path, research_path: Path, now: datetime,
          factor_params: FactorParams = FactorParams(),
          regime_params: RegimeParams = RegimeParams()) -> BuildResult:
    """Build into a fresh temporary file, then atomically replace research_path.

    A fresh file avoids DuckDB file growth from replaced tables, and a failed build leaves the
    previous research database untouched. Build history (feature_builds) is carried over.
    """
    check_warehouse(warehouse_path)
    before = file_hash(warehouse_path)
    started = time.monotonic()
    research_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = research_path.with_name(research_path.stem + ".building.duckdb")
    _remove_db(tmp_path)

    con = duckdb.connect(str(tmp_path))
    try:
        con.execute(f"ATTACH '{warehouse_path.as_posix()}' AS wh (READ_ONLY)")
        con.execute(_BUILDS_DDL)
        if research_path.exists():
            con.execute(f"ATTACH '{research_path.as_posix()}' AS previous (READ_ONLY)")
            has_history = con.execute(
                "SELECT count(*) FROM duckdb_tables() WHERE database_name = 'previous' AND table_name = 'feature_builds'"
            ).fetchone()[0]
            if has_history:
                con.execute("INSERT INTO feature_builds BY NAME SELECT * FROM previous.feature_builds")
            con.execute("DETACH previous")
        con.begin()
        try:
            for _name, sql in sql_files():
                con.execute(sql)
            build_factors(con, factor_params)
            build_regimes(con, regime_params)
            build_cost_inputs(con)
            rows = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                    for t in ("daily_panel", "market_daily", "stock_features", "stock_targets", "stock_factors",
                              "market_regimes", "stock_trading_costs")}
            data_as_of = con.execute("SELECT max(date)::VARCHAR FROM daily_panel").fetchone()[0]
            fetched = con.execute("SELECT max(fetched_at) FROM wh.quotes_daily").fetchone()[0]
            build_id = now.strftime("%Y%m%dT%H%M%S")
            seconds = round(time.monotonic() - started, 2)
            con.execute(
                """INSERT OR REPLACE INTO feature_builds (build_id, feature_set_version, built_at, data_as_of,
                       warehouse_fetched_at, code_hash, panel_rows, feature_rows, target_rows, seconds, warehouse_sha,
                       factor_params, regime_params, factors_version, regimes_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [build_id, FEATURE_SET_VERSION, now, data_as_of, fetched, feature_build_hash(),
                 rows["daily_panel"], rows["stock_features"], rows["stock_targets"], seconds, before,
                 params_json(factor_params), params_json(regime_params), factors_version(), regimes_version()],
            )
            con.commit()
        except BaseException:
            con.rollback()
            raise
        con.execute("DETACH wh")
        con.execute("CHECKPOINT")
    except BaseException:
        con.close()
        _remove_db(tmp_path)
        raise
    con.close()
    os.replace(tmp_path, research_path)
    research_path.with_name(research_path.name + ".wal").unlink(missing_ok=True)

    if file_hash(warehouse_path) != before:
        raise BuildError("Warehouse file changed during the build; this must never happen")
    return BuildResult(build_id, data_as_of, feature_build_hash(), rows, seconds)

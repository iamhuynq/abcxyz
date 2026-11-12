"""Which feature build the ML data comes from, and whether its effective configuration is the approved one
(docs/ml-plan.md, PR #10 third review).

Each `quant build` records in research.duckdb's feature_builds the parameters it actually used for the factors and
regimes, and the definition versions. The effective source version of a table is computed from that row, never
from the code defaults. A row without this metadata (built before 2026-11-12) is unknown and refused, never
treated as default.
"""

import json

import duckdb

from quant_research import factors, regimes
from quant_research.ml import allowlist
from quant_research.ml.contracts import BaselineSpec, ContractError, DatasetSpec

META = ("factor_params", "regime_params", "factors_version", "regimes_version")


def build_source_versions(con: duckdb.DuckDBPyConnection, build_id: str, schema: str = "") -> dict[str, str]:
    """{table: effective source version} for the approved tables of one build. `schema`: "rs." when research.duckdb
    is attached to a results store, "" for a direct connection."""
    db = schema.rstrip(".") or con.execute("SELECT current_database()").fetchone()[0]
    cols = {c for (c,) in con.execute("SELECT column_name FROM duckdb_columns() WHERE database_name = ? "
                                      "AND table_name = 'feature_builds'", [db]).fetchall()}
    if not cols:
        raise ContractError("no feature_builds table: not a research database")
    absent = [c for c in META if c not in cols]
    if absent:                          # a research.duckdb built before 2026-11-12: the old schema
        raise ContractError(f"feature_builds has no {', '.join(absent)}: the configuration of every build is "
                            "unknown (built before 2026-11-12); rebuild with `quant build`")
    cur = con.execute(f"SELECT {', '.join(META)}, feature_set_version, code_hash FROM {schema}feature_builds "
                      "WHERE build_id = ?", [build_id])
    row = cur.fetchone()
    if row is None:
        raise ContractError(f"no feature build {build_id}")
    factor_params, regime_params, factors_v, regimes_v, feature_set, code = row
    # NULL, a non-string or an empty / blank string is unknown provenance: fail closed, never fall back to the current
    missing = [name for name, v in zip((*META, "feature_set_version", "code_hash"), row)
               if not isinstance(v, str) or not v.strip()]
    if missing:
        raise ContractError(f"feature build {build_id} has no {', '.join(missing)}: its configuration is unknown "
                            "(built before 2026-11-12); rebuild with `quant build`")
    try:
        fp = factors.FactorParams(**json.loads(factor_params))
        rp = regimes.RegimeParams(**json.loads(regime_params))
    except (TypeError, ValueError) as exc:
        raise ContractError(f"feature build {build_id}: stored parameters do not match the current fields: {exc}")
    return {"stock_features": f"{feature_set}@{code}",
            "stock_factors": allowlist.factor_source_version(fp, factors_v, code),
            "market_regimes": allowlist.regime_source_version(rp, regimes_v, code)}


def check_build(con: duckdb.DuckDBPyConnection, build_id: str, dataset: DatasetSpec,
                baseline: BaselineSpec | None = None, schema: str = "") -> None:
    """Refuse the build unless every feature of the dataset, and the baseline factor, were produced under their
    approved source versions."""
    effective = build_source_versions(con, build_id, schema)
    bad = [f"{f.name} (approved {f.source_version}, build {effective[f.table]})" for f in dataset.features
           if effective[f.table] != f.source_version]
    if baseline is not None and effective["stock_factors"] != baseline.source_version:
        bad.append(f"baseline {baseline.factor} (approved {baseline.source_version}, build "
                   f"{effective['stock_factors']})")
    if bad:
        raise ContractError(f"feature build {build_id} was not produced under the approved configuration: "
                            + "; ".join(bad))


def build_identity(con: duckdb.DuckDBPyConnection, build_id: str, schema: str = "") -> dict[str, str]:
    """The data identity of a build: warehouse_sha, code_hash, data_as_of. Empty or NULL values fail closed."""
    row = con.execute(f"SELECT warehouse_sha, code_hash, data_as_of FROM {schema}feature_builds WHERE build_id = ?",
                      [build_id]).fetchone()
    if row is None:
        raise ContractError(f"no feature build {build_id}")
    wh, code, as_of = row
    for name, v in (("warehouse_sha", wh), ("code_hash", code)):
        if not isinstance(v, str) or not v.strip():
            raise ContractError(f"feature build {build_id} has no {name}: its data identity is unknown")
    if as_of is None:
        raise ContractError(f"feature build {build_id} has no data_as_of")
    return {"warehouse_sha": wh, "code_hash": code, "data_as_of": str(as_of)}

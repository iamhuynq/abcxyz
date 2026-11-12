"""Effective build configuration (docs/ml-plan.md, PR #10 third review): feature_builds records the factor and
regime parameters actually used, and ML accepts only builds produced under the approved configuration."""

import dataclasses
import json
from datetime import UTC, datetime

import duckdb
import pytest

from quant_research import factors
from quant_research.build import build
from quant_research.factors import FactorParams
from quant_research.ml import allowlist
from quant_research.ml.builds import build_source_versions, check_build
from quant_research.ml.contracts import BaselineSpec, ContractError, DatasetSpec, FeatureSpec, LabelSpec
from quant_research.regimes import RegimeParams
from tests.research.synthetic import build_synthetic_warehouse

NOW = datetime(2026, 11, 12, 9, 0, tzinfo=UTC)
F = FeatureSpec.current
DATASET = DatasetSpec((F("stock_features", "return_5d"), F("stock_factors", "order_flow.rank_pct"),
                       F("market_regimes", "direction")), LabelSpec("fwd_ret_exec_10d", 10),
                      __import__("datetime").date(2010, 1, 4), __import__("datetime").date(2023, 12, 29))
BASELINE = BaselineSpec.current("order_flow")


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    d = tmp_path_factory.mktemp("mlbuild")
    build_synthetic_warehouse(d / "w.duckdb")
    return d / "w.duckdb"


def _build(warehouse, path, **kw):
    build(warehouse, path, NOW, **kw)
    con = duckdb.connect(str(path), read_only=True)
    build_id = con.execute("SELECT build_id FROM feature_builds ORDER BY built_at DESC LIMIT 1").fetchone()[0]
    return con, build_id


def test_a_default_build_records_its_configuration_and_is_accepted(warehouse, tmp_path):
    con, bid = _build(warehouse, tmp_path / "r.duckdb")
    row = con.execute("SELECT factor_params, regime_params, factors_version FROM feature_builds").fetchone()
    assert json.loads(row[0]) == dataclasses.asdict(FactorParams()) and row[2] == factors.version()
    assert json.loads(row[1]) == dataclasses.asdict(RegimeParams())
    versions = build_source_versions(con, bid)
    assert versions == {"stock_features": allowlist.features_source_version(),
                        "stock_factors": allowlist.factor_source_version(),
                        "market_regimes": allowlist.regime_source_version()}
    check_build(con, bid, DATASET, BASELINE)
    con.close()


def test_a_build_with_other_factor_parameters_is_refused(warehouse, tmp_path):
    con, bid = _build(warehouse, tmp_path / "r.duckdb", factor_params=FactorParams(min_stocks=1))
    assert json.loads(con.execute("SELECT factor_params FROM feature_builds").fetchone()[0])["min_stocks"] == 1
    assert build_source_versions(con, bid)["stock_factors"] != allowlist.factor_source_version()
    with pytest.raises(ContractError, match="stock_factors.order_flow.rank_pct"):
        check_build(con, bid, DATASET)
    only_regime = DatasetSpec((F("market_regimes", "direction"),), DATASET.label, DATASET.period_start,
                              DATASET.period_end)
    check_build(con, bid, only_regime)                                    # regimes were built with the defaults
    with pytest.raises(ContractError, match="baseline order_flow"):
        check_build(con, bid, only_regime, BASELINE)
    con.close()


def test_a_build_with_other_regime_parameters_is_refused(warehouse, tmp_path):
    con, bid = _build(warehouse, tmp_path / "r.duckdb", regime_params=RegimeParams(window=100, min_obs=50))
    with pytest.raises(ContractError, match="market_regimes.direction"):
        check_build(con, bid, DATASET)
    con.close()


def test_a_legacy_or_unknown_build_is_refused_never_treated_as_default(warehouse, tmp_path):
    path = tmp_path / "r.duckdb"
    con, bid = _build(warehouse, path)
    con.close()
    with duckdb.connect(str(path)) as rw:
        rw.execute("UPDATE feature_builds SET factor_params = NULL")
    con = duckdb.connect(str(path), read_only=True)
    with pytest.raises(ContractError, match="configuration is unknown"):
        check_build(con, bid, DATASET)
    with pytest.raises(ContractError, match="no feature build"):
        build_source_versions(con, "missing")
    con.close()


def test_rebuilds_keep_history_with_null_metadata_on_legacy_rows(warehouse, tmp_path):
    path = tmp_path / "r.duckdb"
    con, first = _build(warehouse, path)
    con.close()
    with duckdb.connect(str(path)) as rw:                                  # an older row without the new columns
        rw.execute("INSERT INTO feature_builds (build_id, feature_set_version, built_at, code_hash) "
                   "VALUES ('legacy', 'v2', TIMESTAMPTZ '2026-10-01 00:00:00+00', 'x')")
    build(warehouse, path, datetime(2026, 11, 13, 9, 0, tzinfo=UTC))
    with duckdb.connect(str(path), read_only=True) as con:
        rows = dict(con.execute("SELECT build_id, factor_params IS NULL FROM feature_builds").fetchall())
        assert rows["legacy"] is True and rows[first] is False and len(rows) == 3
        with pytest.raises(ContractError, match="configuration is unknown"):
            build_source_versions(con, "legacy")


def test_factors_version_does_not_depend_on_parameters(monkeypatch):
    import functools
    before = factors.version()
    monkeypatch.setattr(factors, "FactorParams", functools.partial(FactorParams, winsor=0.05))
    assert factors.version() == before and allowlist.factor_source_version() != allowlist.factor_source_version(
        FactorParams())


def test_an_old_schema_without_the_columns_is_refused_with_a_contract_error(tmp_path):
    path = tmp_path / "old.duckdb"
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE TABLE feature_builds (build_id VARCHAR, feature_set_version VARCHAR, built_at TIMESTAMPTZ)")
        con.execute("INSERT INTO feature_builds VALUES ('b1', 'v2', TIMESTAMPTZ '2026-11-10 00:00:00+00')")
    with duckdb.connect(str(path), read_only=True) as con:
        with pytest.raises(ContractError, match="configuration of every build is unknown"):
            build_source_versions(con, "b1")
    with duckdb.connect() as con:
        with pytest.raises(ContractError, match="no feature_builds table"):
            build_source_versions(con, "b1")


def test_attached_research_database_through_a_schema(warehouse, tmp_path):
    con, bid = _build(warehouse, tmp_path / "r.duckdb")
    con.close()
    with duckdb.connect() as c:
        c.execute(f"ATTACH '{(tmp_path / 'r.duckdb').as_posix()}' AS rs (READ_ONLY)")
        assert build_source_versions(c, bid, schema="rs.")["stock_features"] == allowlist.features_source_version()


def test_every_source_version_is_bound_to_the_build_code(monkeypatch):
    old = {t: allowlist.source_version(t) for t in allowlist.ALLOWED}
    old_feats = {t: F(t, allowlist.ALLOWED[t][0]) for t in allowlist.ALLOWED}
    old_baseline = BaselineSpec.current("order_flow")
    monkeypatch.setattr(allowlist, "feature_build_hash", lambda: "0123456789abcdef")      # the build code changed
    new = {t: allowlist.source_version(t) for t in allowlist.ALLOWED}
    assert all(new[t] != old[t] and new[t].endswith("@0123456789abcdef") for t in new)
    assert all(new[t].split("@")[0] == old[t].split("@")[0] for t in new)          # definitions and params as before
    for t, f in old_feats.items():
        with pytest.raises(ContractError, match="definition changed"):
            FeatureSpec(f.table, f.column, f.source_version)
    with pytest.raises(ContractError, match="factor definitions changed"):
        BaselineSpec("order_flow", old_baseline.source_version)


def test_a_build_made_by_other_code_is_refused(warehouse, tmp_path):
    path = tmp_path / "r.duckdb"
    con, bid = _build(warehouse, path)
    con.close()
    with duckdb.connect(str(path)) as rw:
        rw.execute("UPDATE feature_builds SET code_hash = 'ffffffffffffffff'")
    with duckdb.connect(str(path), read_only=True) as con:
        assert all(v.endswith("@ffffffffffffffff") for v in build_source_versions(con, bid).values())
        with pytest.raises(ContractError, match="stock_features.return_5d"):
            check_build(con, bid, DATASET)


def test_the_build_code_hash_covers_the_factor_implementation_and_every_build_sql():
    import inspect
    from pathlib import Path
    from quant_research.build import feature_build_files, sql_files
    files = feature_build_files()
    assert Path(inspect.getsourcefile(factors.build_factors)).name == "factors.py" and "factors.py" in files
    assert {"regimes.py", "build.py", "backtest/costs.py"} <= set(files)
    assert {f"sql/{name}" for name, _ in sql_files()} <= set(files)


@pytest.mark.parametrize("column", ["code_hash", "factors_version", "regimes_version", "feature_set_version",
                                    "factor_params", "regime_params"])
@pytest.mark.parametrize("value", ["", "   "])
def test_empty_provenance_is_refused_never_replaced_by_the_current_version(warehouse, tmp_path, column, value):
    path = tmp_path / "r.duckdb"
    con, bid = _build(warehouse, path)
    con.close()
    with duckdb.connect(str(path)) as rw:
        rw.execute(f"UPDATE feature_builds SET {column} = ?", [value])
    with duckdb.connect(str(path), read_only=True) as con:
        with pytest.raises(ContractError, match=f"no .*{column}"):
            build_source_versions(con, bid)


def test_version_helpers_refuse_explicit_empty_values():
    for call in (lambda: allowlist.features_source_version(""), lambda: allowlist.factor_source_version(code=""),
                 lambda: allowlist.factor_source_version(definitions=""), lambda: allowlist.regime_source_version(code=" "),
                 lambda: allowlist.regime_source_version(definitions="")):
        with pytest.raises(ValueError, match="empty or invalid provenance"):
            call()

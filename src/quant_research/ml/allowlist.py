"""Approved ML feature columns (docs/ml-plan.md section 6.2). Explicit: nothing is taken from a table
automatically, and a column not listed here is refused even in an approved table.

Every approved table is classified point-in-time by tests/research/test_integrity.py (every column of it is
unchanged when later data changes). Each source has a version: a feature approved under one definition is
refused once that definition changes.
"""

from quant_research import factors, regimes
from quant_research.build import FEATURE_SET_VERSION, feature_build_hash

AVAILABILITY = "close_t"           # known at the close of the decision session t (decision after the close)

# The stock_features columns the pure-Python reference implementation checks (tests/research/test_feature_build.py).
STOCK_FEATURES = (
    "return_1d", "return_3d", "return_5d", "return_10d", "return_20d", "excess_return_5d",
    "volume_ratio_5", "volume_ratio_20", "volume_change", "volume_zscore_20", "volatility_5", "volatility_20",
    "atr_14_pct", "high_low_range", "gap", "close_position", "close_vs_avg_price", "order_imbalance",
    "volume_imbalance", "buy_pressure", "sell_pressure", "buy_sell_ratio", "foreign_net_value",
    "foreign_net_3d", "foreign_net_5d", "foreign_net_20d", "foreign_net_ratio_20", "foreign_intensity",
    "adv_value_20", "market_regime")
FACTOR_FIELDS = ("rank_pct", "z")
# z_industry is NOT approved: it subtracts the ICB level-2 mean, and the ICB code is the current classification
# (no history; integrity finding E-1), so it is not point in time. Until an industry history exists.
LIMITATIONS = ("z_industry excluded: industry is the current ICB classification, not point in time",
               "exchange and industry are current values (integrity finding E-1)")
STOCK_FACTORS = tuple(f"{f}.{field}" for f in factors.FACTORS for field in FACTOR_FIELDS)
MARKET_REGIMES = tuple(regimes.DIMENSIONS)

ALLOWED: dict[str, tuple[str, ...]] = {"stock_features": STOCK_FEATURES, "stock_factors": STOCK_FACTORS,
                                       "market_regimes": MARKET_REGIMES}

# Sources refused by name, with the reason (anything else that is not in ALLOWED is refused too).
EXCLUDED: dict[str, str] = {
    "stock_targets": "outcomes, never features",
    "feature_target": "joins outcomes to features",
    "report_marks": "no publication time",
    "fundamental_snapshots": "no publication time",
    "interaction_tests": "Interaction Engine scan statistics, not per-row values",
    "interaction_stats": "Interaction Engine scan statistics, not per-row values",
    "exposure": "Exposure Engine output is portfolio-level",
    "risk_positions": "Risk Engine output is portfolio-level",
    "risk_metrics": "Risk Engine output is portfolio-level",
    "cross_features": "cross.duckdb is not approved for ML in PR #10",
    "daily_panel": "raw panel (back-adjusted levels, flags): use the derived features",
}


# Every source version ends with "@<feature build code hash>": feature_build_hash() covers every build SQL file and
# the whole of build.py, factors.py (build_factors included), regimes.py and backtest/costs.py, i.e. all the code
# that produces research.duckdb, upstream tables included. The approved version uses the current code; a build's
# effective version uses its stored feature_builds.code_hash (ml/builds.py). Coarse on purpose: any change to the
# build code needs a rebuild before ML accepts the data.

def _given(value: str | None, current) -> str:
    """The supplied value; the current one only when the argument is omitted (None). An empty string is refused,
    never replaced by the current version."""
    if value is None:
        return current()
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"empty or invalid provenance value {value!r}")
    return value


def features_source_version(code: str | None = None) -> str:
    """The version of stock_features: the feature set and the build code."""
    return f"{FEATURE_SET_VERSION}@{_given(code, feature_build_hash)}"


def factor_source_version(params=None, definitions: str | None = None, code: str | None = None) -> str:
    """The version of stock_factors: the factor set, the definitions (factors.version(): SQL and factors), a
    canonical hash of every FactorParams field and the build code. Defaults: the current definitions, default
    parameters and current code, i.e. the approved version used by FeatureSpec and BaselineSpec. factors.version()
    is left unchanged: recorded interaction runs store it."""
    from quant_research.ml.canonical import digest
    params = factors.FactorParams() if params is None else params
    return f"{factors.FACTOR_SET}:{_given(definitions, factors.version)}:{digest(params)[:8]}@{_given(code, feature_build_hash)}"


def regime_source_version(params=None, definitions: str | None = None, code: str | None = None) -> str:
    """The version of market_regimes: the definitions (regimes.version()), a canonical hash of every RegimeParams
    field and the build code; defaults as for factor_source_version."""
    from quant_research.ml.canonical import digest
    params = regimes.RegimeParams() if params is None else params
    return f"{_given(definitions, regimes.version)}:{digest(params)[:8]}@{_given(code, feature_build_hash)}"


def source_version(table: str) -> str:
    if table == "stock_features":
        return features_source_version()
    if table == "stock_factors":
        return factor_source_version()
    if table == "market_regimes":
        return regime_source_version()
    raise KeyError(table)


def refusal(table: str, column: str, version: str, availability: str) -> str:
    """Why this feature is not allowed ("" if it is)."""
    if table in EXCLUDED:
        return f"source {table} is excluded: {EXCLUDED[table]}"
    if table not in ALLOWED:
        return f"source {table} is not an approved point-in-time source"
    if column not in ALLOWED[table]:
        return f"column {column} of {table} is not in the allowlist"
    if availability != AVAILABILITY:
        return f"availability {availability!r} is not {AVAILABILITY!r} (decision at the close of t)"
    current = source_version(table)
    if version != current:
        return f"source version {version!r} of {table} is not the current {current!r}: the definition changed"
    return ""

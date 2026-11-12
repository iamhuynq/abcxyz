"""Regime Engine (docs/regime-interaction-plan.md, part 1): point-in-time market regime labels per session.

Built inside `quant build` into `market_regimes`. Continuous dimensions are labelled by the tercile of today's
value within the trailing window of sessions (today included); no threshold is tuned. The risk composite is
the rule declared in the plan before any outcome was looked at.
"""

import hashlib
import inspect
import json
from dataclasses import dataclass

import duckdb

DIMENSIONS = {   # name: (states low -> high, underlying value)
    "direction": (("Bear", "Sideway", "Bull"), "VNINDEX vs MA50 / MA200 (market_daily.market_regime)"),
    "volatility": (("low", "normal", "high"), "VNINDEX 20-session volatility"),
    "liquidity": (("low", "normal", "high"), "total deal_value of all stocks, 20-session mean"),
    "breadth": (("weak", "neutral", "strong"), "share of liquid stocks closing above their own 50-session mean"),
    "foreign": (("selling", "neutral", "buying"), "market foreign net value / turnover, 20 sessions"),
    "risk": (("risk_off", "neutral", "risk_on"), "composite rule (see risk_sql)"),
}
TERCILE_DIMENSIONS = ("volatility", "liquidity", "breadth", "foreign")


@dataclass(frozen=True)
class RegimeParams:
    window: int = 500                  # trailing sessions for the tercile
    min_obs: int = 250                 # label is NULL before this many values
    breadth_min_adv: float = 1e9       # liquid stocks for the breadth measure
    drawdown_off: float = -0.20        # risk_off when VNINDEX is this far below its 250-session high


VALUES_SQL = """
CREATE OR REPLACE TEMP TABLE regime_values AS
WITH s AS (
    SELECT symbol, date, adj_close, is_traded, deal_value, bad_source_date, foreign_inconsistent, price_jump,
           buy_foreign_value - sell_foreign_value AS foreign_net,
           CASE WHEN count(adj_close) OVER w50 = 50 THEN avg(adj_close) OVER w50 END AS ma50
    FROM daily_panel
    WINDOW w50 AS (PARTITION BY symbol ORDER BY date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW)
),
d AS (
    SELECT s.date,
           sum(s.deal_value) FILTER (WHERE s.is_traded AND NOT s.bad_source_date) AS turnover,
           count(*) FILTER (WHERE s.is_traded AND NOT s.bad_source_date) AS n_traded,
           sum(s.foreign_net) FILTER (WHERE s.is_traded AND NOT s.bad_source_date AND NOT s.foreign_inconsistent)
               AS foreign_net,
           avg((s.adj_close > s.ma50)::INT) FILTER (WHERE s.is_traded AND NOT s.price_jump AND NOT s.bad_source_date
               AND f.adv_value_20 > {breadth_min_adv} AND s.ma50 IS NOT NULL) AS breadth
    FROM s JOIN stock_features f USING (symbol, date)
    GROUP BY s.date
),
m AS (
    SELECT m.date, m.market_regime, m.mkt_vol_20 AS volatility, m.mkt_close,
           CASE WHEN d.n_traded > 0 THEN d.turnover END AS turnover, d.foreign_net, d.breadth
    FROM market_daily m LEFT JOIN d USING (date)
)
SELECT date, market_regime, volatility, breadth,
       CASE WHEN count(turnover) OVER w20 >= 15 THEN avg(turnover) OVER w20 END AS liquidity,
       CASE WHEN count(turnover) OVER w20 >= 15
            THEN sum(foreign_net) OVER w20 / nullif(sum(turnover) OVER w20, 0) END AS foreign,
       mkt_close / max(mkt_close) OVER (ORDER BY date ROWS BETWEEN 249 PRECEDING AND CURRENT ROW) - 1 AS drawdown,
       row_number() OVER (ORDER BY date) AS i
FROM m
WINDOW w20 AS (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW);
"""


def _percentile_sql(dim: str, window: int, min_obs: int) -> str:
    """Share of the trailing window's non-NULL values that are <= today's value (today included)."""
    return f"""
        SELECT a.date, CASE WHEN count(b.{dim}) >= {min_obs}
                            THEN count(*) FILTER (WHERE b.{dim} <= a.{dim}) / count(b.{dim}) END AS pct
        FROM regime_values a JOIN regime_values b ON b.i BETWEEN a.i - {window - 1} AND a.i
        WHERE a.{dim} IS NOT NULL GROUP BY a.date"""


def _state_sql(dim: str) -> str:
    low, mid, high = DIMENSIONS[dim][0]
    p = f"{dim}_pct"
    return (f"CASE WHEN {p} IS NULL THEN NULL WHEN {p} <= 1.0 / 3 THEN '{low}' "
            f"WHEN {p} > 2.0 / 3 THEN '{high}' ELSE '{mid}' END AS {dim}")


def build_regimes(con: duckdb.DuckDBPyConnection, params: RegimeParams = RegimeParams()) -> int:
    con.execute(VALUES_SQL.format(breadth_min_adv=float(params.breadth_min_adv)))
    joins = " ".join(f"LEFT JOIN ({_percentile_sql(d, int(params.window), int(params.min_obs))}) p_{d} USING (date)"
                     for d in TERCILE_DIMENSIONS)
    pcts = ", ".join(f"p_{d}.pct AS {d}_pct" for d in TERCILE_DIMENSIONS)
    con.execute(f"""
        CREATE OR REPLACE TABLE market_regimes AS
        WITH p AS (SELECT v.date, v.market_regime, v.volatility AS volatility_value, v.liquidity AS liquidity_value,
                          v.breadth AS breadth_value, v.foreign AS foreign_value, v.drawdown, {pcts}
                   FROM regime_values v {joins}),
        l AS (SELECT p.*, p.market_regime AS direction, {", ".join(_state_sql(d) for d in TERCILE_DIMENSIONS)}
              FROM p)
        SELECT * EXCLUDE (market_regime),
               CASE WHEN volatility IS NULL OR breadth IS NULL OR direction IS NULL THEN NULL
                    WHEN (volatility = 'high' AND breadth = 'weak') OR drawdown <= {float(params.drawdown_off)}
                         THEN 'risk_off'
                    WHEN volatility <> 'high' AND breadth = 'strong' AND direction = 'Bull' THEN 'risk_on'
                    ELSE 'neutral' END AS risk
        FROM l ORDER BY date""")
    con.execute("DROP TABLE regime_values")
    return con.execute("SELECT count(*) FROM market_regimes").fetchone()[0]


def version() -> str:
    """Hash of the regime **definitions**: the values SQL, the percentile, state and build code (the risk rule lives
    in build_regimes), DIMENSIONS and TERCILE_DIMENSIONS. Canonical JSON (sorted keys), so dictionary order and
    object repr() do not matter. The parameters (RegimeParams) are versioned separately, like factors.version():
    each build records the ones it used (feature_builds.regime_params)."""
    payload = {"values_sql": VALUES_SQL,
               "code": [inspect.getsource(f) for f in (_percentile_sql, _state_sql, build_regimes)],
               "dimensions": {k: [list(states), what] for k, (states, what) in DIMENSIONS.items()},
               "tercile_dimensions": list(TERCILE_DIMENSIONS)}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()[:8]

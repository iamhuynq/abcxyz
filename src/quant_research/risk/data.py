"""Inputs of a risk evaluation (read-only). Every query names its schema: `rs.` is research.duckdb attached to a
results store, "" is a direct research connection (the UI), `cx.` is cross.duckdb when attached.

A snapshot is a portfolio at the close of one valuation date: positions (shares, price, value), cash, and the
equity curves that end on that date. Prices are the portfolio's own marks.
"""

from dataclasses import dataclass, field
from datetime import date

import duckdb
import numpy as np
import pandas as pd

from quant_research.paper_state import current_paper_date

EXPOSURE_SESSIONS = 120          # the Exposure panel's window (stock_ui), reused unchanged


def exposure_returns(con: duckdb.DuckDBPyConnection, symbols: tuple[str, ...], as_of: date | None = None,
                     schema: str = "", sessions: int = EXPOSURE_SESSIONS) -> dict:
    """Clean daily returns (traded, no price jump, not the bad source date, previous session clean) of
    `symbols` over the `sessions` market sessions ending at `as_of` (default: the latest), and the market return.
    Moved from stock_ui/exposure_data.py; with as_of None it is the same query."""
    marks = ", ".join("?" for _ in symbols)
    cut, bind = ("WHERE date <= ?", [as_of]) if as_of is not None else ("", [])
    start = con.execute(f"""SELECT min(date) FROM (SELECT date FROM {schema}market_daily {cut}
                            ORDER BY date DESC LIMIT {sessions + 1})""", bind).fetchone()[0]
    end_sql, end_bind = ("AND date <= ?", [as_of]) if as_of is not None else ("", [])
    stocks = con.execute(f"""
        WITH p AS (SELECT symbol, date, adj_close, is_traded AND NOT price_jump AND NOT bad_source_date AS ok,
                          lag(adj_close) OVER w AS prev_close,
                          lag(is_traded AND NOT price_jump AND NOT bad_source_date) OVER w AS prev_ok
                   FROM {schema}daily_panel WHERE symbol IN ({marks}) AND date >= ? {end_sql}
                   WINDOW w AS (PARTITION BY symbol ORDER BY date))
        SELECT symbol, date, CASE WHEN ok AND prev_ok AND prev_close > 0 THEN adj_close / prev_close - 1 END AS r
        FROM p WHERE date > ?""", [*symbols, start, *end_bind, start]).df()
    market = con.execute(f"SELECT date, mkt_ret_1d FROM {schema}market_daily WHERE date > ? {end_sql} ORDER BY date",
                         [start, *end_bind]).df()
    wide = stocks.pivot(index="date", columns="symbol", values="r") if not stocks.empty else pd.DataFrame()
    dates = pd.Index(market["date"])
    return {"returns": wide.reindex(dates), "market": market.set_index("date")["mkt_ret_1d"],
            "as_of": market["date"].max()}


@dataclass
class Snapshot:
    target: str                                  # "paper:<run_date>" or "backtest:<run_id>"
    valuation_date: date
    nav: float                                   # equity at the close, the portfolio's default marks
    cash: float
    positions: pd.DataFrame                      # symbol, shares, price, value
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)   # name -> (dates, equity)
    notes: list[str] = field(default_factory=list)
    reconciliation_gap: float | None = None      # NAV - cash - positions; None when cash is derived from the NAV


def paper_snapshot(con: duckdb.DuckDBPyConnection) -> Snapshot | None:
    """The current paper record (`paper_state`): its positions and both of its equity curves."""
    run_date = current_paper_date(con)
    if run_date is None:
        return None
    daily = con.execute("""SELECT date, equity, equity_traded_mark FROM paper_daily WHERE run_date = ?
                           ORDER BY date""", [run_date]).df()
    if daily.empty:
        return None
    pos = con.execute("""SELECT symbol, shares, last_price AS price, shares * last_price AS value FROM paper_positions
                         WHERE run_date = ? AND shares > 0 ORDER BY symbol""", [run_date]).df()
    nav = float(daily["equity"].iloc[-1])
    dates = daily["date"].to_numpy()
    curves = {"pre-registered marks": (dates, daily["equity"].to_numpy(float))}
    if daily["equity_traded_mark"].notna().all():
        curves["traded marks"] = (dates, daily["equity_traded_mark"].to_numpy(float))
    notes = ["Positions are valued at the pre-registered marks (last close, legacy mode).",
             "The paper record stores no cash: cash is NAV minus positions, so no reconciliation is possible."]
    day = pd.Timestamp(daily["date"].iloc[-1]).date()
    return Snapshot(f"paper:{run_date}", day, nav, nav - float(pos["value"].sum(min_count=1) or 0.0), pos, curves,
                    notes)


def backtest_snapshot(con: duckdb.DuckDBPyConnection, run_id: str, on: date | None = None) -> Snapshot | None:
    """A stored backtest run at the close of `on` (default: its last session). Held positions are rebuilt from
    its trades (entered on or before `on`, not yet exited at its close) and valued at their last traded adjusted
    close on or before `on`; the gap to the stored equity is reported as a note."""
    daily = con.execute("SELECT date, equity, cash FROM backtest_daily WHERE run_id = ? AND (? IS NULL OR date <= ?) "
                        "ORDER BY date", [run_id, on, on]).df()
    if daily.empty:
        return None
    day = pd.Timestamp(daily["date"].iloc[-1]).date()
    held = con.execute("""
        SELECT t.symbol, sum(t.shares) AS shares FROM backtest_trades t
        WHERE t.run_id = ? AND t.entry_date <= ? AND (t.exit_date > ? OR t.exit_reason = 'open_at_end')
        GROUP BY 1 ORDER BY 1""", [run_id, day, day]).df()
    prices = _last_traded_close(con, tuple(held["symbol"]), day)
    held["price"] = held["symbol"].map(prices)
    held["value"] = held["shares"] * held["price"]
    nav, cash = float(daily["equity"].iloc[-1]), float(daily["cash"].iloc[-1])
    gap = nav - cash - float(held["value"].sum())              # NaN when a position has no price
    gap = 0.0 if abs(gap) < 0.5 else gap                       # below half a dong: rounding, printed as 0
    notes = [f"Positions rebuilt from the stored trades; positions + cash differ from the stored equity by "
             f"{gap:,.0f} VND ({gap / nav:.4%})."]
    curves = {"stored equity": (daily["date"].to_numpy(), daily["equity"].to_numpy(float))}
    return Snapshot(f"backtest:{run_id}", day, nav, cash, held, curves, notes, gap)


def _last_traded_close(con: duckdb.DuckDBPyConnection, symbols: tuple[str, ...], on: date) -> dict[str, float]:
    if not symbols:
        return {}
    marks = ", ".join("?" for _ in symbols)
    return dict(con.execute(f"""SELECT symbol, arg_max(adj_close, date) FROM rs.daily_panel
                                WHERE symbol IN ({marks}) AND date <= ? AND is_traded GROUP BY 1""",
                            [*symbols, on]).fetchall())


def market_inputs(con: duckdb.DuckDBPyConnection, symbols: tuple[str, ...], on: date) -> pd.DataFrame:
    """Per symbol at the close of `on`: ADV20, half-spread and sigma (cost inputs of that close), the ICB level-2
    code, last traded date, sessions since it, and whether it closed at the floor. The ICB code is the CURRENT
    classification: daily_panel copies it from today's symbol table on every row (no history; integrity
    finding E-1), so it is not point in time."""
    cols = ["symbol", "adv", "half_spread", "sigma", "industry", "last_trade", "stale_sessions", "at_floor"]
    if not symbols:
        return pd.DataFrame(columns=cols)
    marks = ", ".join("?" for _ in symbols)
    return con.execute(f"""
        WITH s AS (SELECT unnest([{marks}]) AS symbol),
        last AS (SELECT symbol, max(date) AS last_trade FROM rs.daily_panel
                 WHERE symbol IN ({marks}) AND date <= ? AND is_traded GROUP BY 1),
        cal AS (SELECT date FROM rs.market_daily WHERE date <= ?),
        today AS (SELECT symbol, industry_l2_code, limit_down FROM rs.daily_panel WHERE date = ?),
        ind AS (SELECT symbol, any_value(industry_l2_code) AS industry_l2_code FROM rs.daily_panel
                WHERE symbol IN ({marks}) AND date <= ? GROUP BY 1)                 -- current classification
        SELECT s.symbol, c.adv_value_20 AS adv, c.half_spread, c.sigma_20 AS sigma, ind.industry_l2_code AS industry,
               last.last_trade,
               (SELECT count(*) FROM cal WHERE cal.date > last.last_trade) AS stale_sessions,
               coalesce(today.limit_down, false) AS at_floor
        FROM s LEFT JOIN rs.stock_trading_costs c ON c.symbol = s.symbol AND c.date = ?
        LEFT JOIN last ON last.symbol = s.symbol LEFT JOIN today ON today.symbol = s.symbol
        LEFT JOIN ind ON ind.symbol = s.symbol
        ORDER BY s.symbol""", [*symbols, *symbols, on, on, on, *symbols, on, on]).df()[cols]


def clusters(con: duckdb.DuckDBPyConnection, symbols: tuple[str, ...], on: date) -> dict[str, str]:
    """Cluster label of each symbol at the last month end on or before `on` (point in time); {} without cx."""
    has = con.execute("SELECT count(*) FROM duckdb_databases() WHERE database_name = 'cx'").fetchone()[0]
    if not has or not symbols:
        return {}
    marks = ", ".join("?" for _ in symbols)
    rows = con.execute(f"""
        WITH m AS (SELECT max(month_end) AS d FROM cx.clusters_monthly WHERE month_end <= ?)
        SELECT s.symbol, 'cluster ' || c.cluster FROM cx.clusters_monthly c
        JOIN m ON c.month_end = m.d JOIN cx.symbols s ON s.id = c.symbol_id
        WHERE s.symbol IN ({marks})""", [on, *symbols]).fetchall()
    return dict(rows)


def factor_z(con: duckdb.DuckDBPyConnection, on: date) -> pd.DataFrame:
    """Factor z-scores (symbols x factors) on the last scored date on or before `on` (display only)."""
    df = con.execute("""SELECT symbol, factor, z FROM rs.stock_factors
                        WHERE date = (SELECT max(date) FROM rs.stock_factors WHERE date <= ?)""", [on]).df()
    return df.pivot(index="symbol", columns="factor", values="z") if not df.empty else pd.DataFrame()

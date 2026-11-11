"""Load the research panel into dense (date x symbol) matrices for the backtest.

Everything here is point-in-time: the universe and the signal on date d use only data known at the
close of d. Forward-looking Phase 3 filters (fwd_has_price_jump_20d, crosses_period, entry_blocked)
are deliberately not used.
"""

from dataclasses import dataclass
from datetime import date

import duckdb
import numpy as np

from quant_research.patterns import DECILE_TIE_BREAK


@dataclass(frozen=True)
class UniverseRule:
    min_adv_value: float = 1e9
    min_session_index: int = 20
    min_stocks_per_date: int = 30
    signal_feature: str = "order_imbalance"
    signal_decile: int = 10

    def sql(self, prefix: str = "") -> str:
        c = prefix
        return (f"{c}is_traded AND {c}adv_value_20 > {self.min_adv_value} AND {c}session_index > {self.min_session_index}"
                f" AND NOT {c}price_jump AND NOT {c}bad_source_date")


@dataclass
class Market:
    dates: np.ndarray              # datetime64[D], trading calendar (VNINDEX sessions)
    symbols: np.ndarray            # str
    open: np.ndarray               # adjusted open, NaN when no row
    close: np.ndarray              # adjusted close, NaN when no row
    traded: np.ndarray             # bool
    open_limit_up: np.ndarray      # bool, cannot buy at this open
    limit_down: np.ndarray         # bool, cannot sell at this close
    price_jump: np.ndarray         # bool, data-error move on this day
    adv_value: np.ndarray          # VND, NaN when unknown (capacity)
    universe: np.ndarray           # bool, point-in-time universe on that date
    signal: np.ndarray             # float: signal value if in the signal decile on that date, else NaN
    index_close: np.ndarray        # VNINDEX close per date
    regime: np.ndarray | None = None  # market_regime per date (str or None)
    half_spread: np.ndarray | None = None  # cost model v1 inputs at that close (stock_trading_costs)
    sigma: np.ndarray | None = None
    tick_half_spread: np.ndarray | None = None

    def date_index(self, d: date) -> int:
        return int(np.searchsorted(self.dates, np.datetime64(d, "D")))


def load_market(con: duckdb.DuckDBPyConnection, start: date, end: date, rule: UniverseRule,
                schema: str = "rs") -> Market:
    """`con` must have the research database attached as `schema` (read-only)."""
    cal = con.execute(f"""SELECT date, mkt_close, market_regime FROM {schema}.market_daily
                          WHERE date BETWEEN ? AND ? ORDER BY date""", [start, end]).fetchall()
    dates = np.array([r[0] for r in cal], dtype="datetime64[D]")
    index_close = np.array([r[1] for r in cal], dtype=float)
    regime = np.array([r[2] for r in cal], dtype=object)
    f = rule.signal_feature
    rows = con.execute(f"""
        WITH base AS (
            SELECT p.symbol, p.date, p.adj_open, p.adj_close, p.is_traded, p.open_limit_up, p.limit_down,
                   p.price_jump, s.adv_value_20, s.{f} AS sig,
                   ({rule.sql("s.")}) AS in_universe
            FROM {schema}.daily_panel p JOIN {schema}.stock_features s USING (symbol, date)
            WHERE p.date BETWEEN ? AND ?
        ),
        ranked AS (
            SELECT *, CASE WHEN in_universe AND sig IS NOT NULL AND isfinite(sig) THEN
                        ntile(10) OVER (PARTITION BY date, (in_universe AND sig IS NOT NULL AND isfinite(sig)) ORDER BY sig, {DECILE_TIE_BREAK})
                      END AS decile,
                   count(*) FILTER (WHERE in_universe AND sig IS NOT NULL AND isfinite(sig)) OVER (PARTITION BY date) AS n_ranked
            FROM base
        )
        SELECT symbol, date, adj_open, adj_close, is_traded, open_limit_up, limit_down, price_jump, adv_value_20,
               coalesce(in_universe, FALSE),
               CASE WHEN decile = {rule.signal_decile} AND n_ranked >= {rule.min_stocks_per_date} THEN sig END
        FROM ranked ORDER BY symbol, date""", [start, end]).fetchall()

    symbols = np.array(sorted({r[0] for r in rows}), dtype=object)
    s_index = {s: i for i, s in enumerate(symbols)}
    shape = (len(dates), len(symbols))
    m = Market(dates, symbols,
               np.full(shape, np.nan), np.full(shape, np.nan), np.zeros(shape, bool), np.zeros(shape, bool),
               np.zeros(shape, bool), np.zeros(shape, bool), np.full(shape, np.nan), np.zeros(shape, bool),
               np.full(shape, np.nan), index_close, regime)
    d_index = {d: i for i, d in enumerate(dates.tolist())}
    for sym, d, o, c, traded, olu, ld, jump, adv, uni, sig in rows:
        i = d_index.get(d)
        if i is None:                     # session missing from the VNINDEX calendar: ignored
            continue
        j = s_index[sym]
        m.open[i, j], m.close[i, j] = o, c
        m.traded[i, j], m.open_limit_up[i, j] = bool(traded), bool(olu)
        m.limit_down[i, j], m.price_jump[i, j] = bool(ld), bool(jump)
        m.adv_value[i, j] = adv if adv is not None else np.nan
        m.universe[i, j] = bool(uni)
        m.signal[i, j] = sig if sig is not None else np.nan
    _load_costs(con, m, start, end, schema, d_index, s_index)
    return m


def _load_costs(con: duckdb.DuckDBPyConnection, m: Market, start: date, end: date, schema: str,
                d_index: dict, s_index: dict) -> None:
    has = con.execute("SELECT count(*) FROM duckdb_tables() WHERE database_name = ? AND table_name = ?",
                      [schema, "stock_trading_costs"]).fetchone()[0]
    if not has:
        return
    m.half_spread = np.full(m.open.shape, np.nan)
    m.sigma = np.full(m.open.shape, np.nan)
    m.tick_half_spread = np.full(m.open.shape, np.nan)
    df = con.execute(f"""SELECT symbol, date, half_spread, sigma_20, tick_half_spread FROM {schema}.stock_trading_costs
                         WHERE date BETWEEN ? AND ?""", [start, end]).df()
    if df.empty:
        return
    i = np.array([d_index.get(d, -1) for d in df["date"].dt.date], dtype=int)
    j = np.array([s_index.get(x, -1) for x in df["symbol"]], dtype=int)
    ok = (i >= 0) & (j >= 0)
    m.half_spread[i[ok], j[ok]] = df["half_spread"].to_numpy(float, na_value=np.nan)[ok]
    m.sigma[i[ok], j[ok]] = df["sigma_20"].to_numpy(float, na_value=np.nan)[ok]
    m.tick_half_spread[i[ok], j[ok]] = df["tick_half_spread"].to_numpy(float, na_value=np.nan)[ok]

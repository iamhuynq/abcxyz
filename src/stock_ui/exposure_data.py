"""Data for the Exposure panel (read-only, bound parameters): returns, factor z-scores (Factor Engine),
labels, paper weights."""

import duckdb
import pandas as pd

from quant_research.risk.data import exposure_returns
from stock_ui.db import Reader, ReadResult

SESSIONS = 120


def _returns(con: duckdb.DuckDBPyConnection, symbols: tuple[str, ...]) -> dict:
    return exposure_returns(con, symbols, sessions=SESSIONS)      # moved to quant_research (Risk Engine uses it)


def returns(reader: Reader, symbols: tuple[str, ...]) -> ReadResult:
    if not symbols:
        return ReadResult(None, None, False, "no symbols")
    return reader.read("research", ("exposure_returns", symbols), lambda con: _returns(con, symbols))


def _factor_z(con: duckdb.DuckDBPyConnection) -> dict | None:
    """Factor z-scores (symbols x factors) and the 250-session beta on the latest scored date (stock_factors)."""
    has = con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'stock_factors'").fetchone()[0]
    if not has:
        return None
    df = con.execute("""SELECT symbol, factor, z, value FROM stock_factors
                        WHERE date = (SELECT max(date) FROM stock_factors)""").df()
    if df.empty:
        return None
    z = df.pivot(index="symbol", columns="factor", values="z")
    beta = df[df["factor"] == "beta"].set_index("symbol")["value"]
    return {"z": z, "beta_250": beta}


def factor_z(reader: Reader) -> ReadResult:
    return reader.read("research", "exposure_factor_z", _factor_z)


def industries(reader: Reader) -> ReadResult:
    return reader.query("warehouse", "SELECT symbol, industry_l2 FROM symbol_industry")


def clusters(reader: Reader) -> ReadResult:
    return reader.query("cross", """
        WITH last AS (SELECT max(month_end) AS d FROM clusters_monthly),
        main AS (SELECT cluster, mode(industry_l2_code) AS code, count(*) AS n FROM clusters_monthly, last
                 WHERE month_end = last.d GROUP BY 1)
        SELECT s.symbol, 'cluster ' || c.cluster || ' (' || main.n || ' stocks)' AS label
        FROM clusters_monthly c JOIN last ON c.month_end = last.d JOIN symbols s ON s.id = c.symbol_id
        JOIN main USING (cluster)""")


def paper_weights(reader: Reader) -> ReadResult:
    """Open positions of the latest paper-portfolio re-simulation, weighted by market value."""
    def run(con: duckdb.DuckDBPyConnection) -> dict | None:
        has = con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = 'paper_positions'").fetchone()[0]
        if not has:
            return None
        from quant_research.paper_state import current_paper_date
        current = current_paper_date(con)
        if current is None:
            return {}
        rows = con.execute("""SELECT symbol, shares * last_price FROM paper_positions WHERE run_date = ?""",
                           [current]).fetchall()
        return {s: float(v) for s, v in rows if v}
    return reader.read("results", "paper_weights", run)

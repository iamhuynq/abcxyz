"""Data for the Risk page: stored Risk Engine runs only (read-only; nothing is recomputed here)."""

import duckdb
import pandas as pd

from stock_ui.db import Reader, ReadResult


def _has(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = ?", [table]).fetchone()[0])


def _runs(con: duckdb.DuckDBPyConnection) -> pd.DataFrame | None:
    """Every run of kind `risk`, newest first, with its evaluation when one was stored (a failed or running run
    has none)."""
    if not _has(con, "research_runs"):
        return None
    joined = "LEFT JOIN risk_evaluations e USING (run_id)" if _has(con, "risk_evaluations") else ""
    cols = ("e.target, e.valuation_date, e.policy_id || '/' || e.policy_version AS policy, e.model_version, "
            "e.status AS risk_status") if joined else \
        "NULL AS target, NULL AS valuation_date, NULL AS policy, NULL AS model_version, NULL AS risk_status"
    df = con.execute(f"""SELECT r.run_id, r.created_at, r.status AS run_status, r.invalid_reason, {cols}
                         FROM research_runs r {joined} WHERE r.kind = 'risk' ORDER BY r.created_at DESC""").df()
    return df if not df.empty else None


def runs(reader: Reader) -> ReadResult:
    return reader.read("results", "risk_runs", _runs)


def _run(con: duckdb.DuckDBPyConnection, run_id: str) -> dict | None:
    if not _has(con, "risk_evaluations"):
        return None
    ev = con.execute("SELECT * FROM risk_evaluations WHERE run_id = ?", [run_id]).df()
    if ev.empty:
        return None
    q = lambda table, order: (con.execute(f"SELECT * EXCLUDE (run_id) FROM {table} WHERE run_id = ? ORDER BY {order}",
                                          [run_id]).df() if _has(con, table) else pd.DataFrame())
    scen = con.execute("SELECT params::JSON ->> '$.extra.scenarios' FROM research_runs WHERE run_id = ?",
                       [run_id]).fetchone()
    return {"evaluation": ev.iloc[0].to_dict(), "scenarios": scen[0] if scen else None,
            "limits": q("risk_limit_results", "CASE status WHEN 'INCOMPLETE' THEN 0 WHEN 'BREACH' THEN 1 "
                                              "WHEN 'WARNING' THEN 2 ELSE 3 END, limit_name"),
            "metrics": q("risk_metrics", "scope, metric"),
            "positions": q("risk_positions", "weight DESC, symbol"),
            "stress": q("risk_stress_results", "seq"),
            "reconciliation": q("risk_reconciliation", "metric")}


def run(reader: Reader, run_id: str) -> ReadResult:
    return reader.read("results", ("risk_run", run_id), lambda con: _run(con, run_id))

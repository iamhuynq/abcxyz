"""Which paper-portfolio record is current (shared by the UI and the Risk Engine; read-only)."""

import duckdb


def _has(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = ?", [table]).fetchone()[0])


def current_paper_date(con: duckdb.DuckDBPyConnection):
    """The paper record `quant daily` marked as current. Three states:
    - state row present: its paper_date, including an explicit NULL (no valid paper record right now);
    - no state row (no paper_state table, or a run that failed after creating it): the latest run_date, as
      before paper_state existed;
    - no paper table: None."""
    if _has(con, "paper_state"):
        row = con.execute("SELECT paper_date FROM paper_state WHERE id = 1").fetchone()
        if row is not None:
            return row[0]
    if not _has(con, "paper_daily"):
        return None
    return con.execute("SELECT max(run_date) FROM paper_daily").fetchone()[0]

"""A synthetic warehouse wide enough for walk-forward runs (PR #13): N_STOCKS ordinary stocks over N_SESSIONS
sessions, built through the normal pipeline (features, targets, factors), so every fold can reach 30 stocks per
date and 60 dates per test window. Deterministic."""

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from fireant_crawler.store.warehouse import Warehouse
from quant_research.build import build
from tests.research.synthetic import NOW, _series

N_STOCKS, N_SESSIONS = 45, 520
START = date(2021, 1, 4)


def sessions(start: date = START, n: int = N_SESSIONS) -> list[date]:
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def build_wide(d: Path) -> Path:
    days = sessions()
    with Warehouse(d / "w.duckdb") as wh:
        wh.ensure_schema_for_write()
        wh.upsert("icb_industries", [{"industry_code": c, "level": lv, "name": n, "description": None, "fetched_at": NOW}
                                     for c, lv, n in (("50", 1, "L1"), ("5020", 2, "L2"), ("502060", 3, "L3"),
                                                      ("50206030", 4, "L4"))])
        names = [f"S{i:02d}" for i in range(N_STOCKS)]
        wh.upsert("symbols", [{"symbol": s, "fetched_at": NOW, "name": f"CTCP {s}", "exchange": "HSX", "type": "stock",
                               "is_listing": True, "industry_code": None, "icb_code": "50206030", "source": "search"}
                              for s in names]
                  + [{"symbol": "VNINDEX", "fetched_at": NOW, "name": "VNINDEX", "exchange": None, "type": "index",
                      "is_listing": True, "industry_code": None, "icb_code": None, "source": "search"}])
        quotes = [q for i, s in enumerate(names) for q in _series(s, days, 100 + i, 20.0 + i, 0.07)]
        wh.upsert("quotes_daily", quotes + _series("VNINDEX", days, 5, 1200.0, 0.0))
    build(d / "w.duckdb", d / "r.duckdb", datetime(2026, 11, 12, 9, 0, tzinfo=UTC))
    return d / "r.duckdb"

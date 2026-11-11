"""Research integrity invariants (docs/research-integrity.md): every table of research.duckdb is classified as
point in time or outcome, and

- changing data after a cutoff changes no point-in-time row at or before it (all tables, all columns);
- a vendor restatement of past adjustment ratios (what a new corporate action after the cutoff does) changes
  only the back-adjusted price levels, never a feature, factor, event, cost input or outcome;
- one symbol's data never changes another symbol's own time series or outcomes;
- outcome windows that run past a symbol's last session are NULL, never partial;
- the period boundaries are the same everywhere they are written.
"""

import json
import re
from datetime import UTC, date, datetime
from importlib import resources
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from fireant_crawler.store.warehouse import Warehouse
from quant_research.build import build
from quant_research.factors import FactorParams
from quant_research.regimes import RegimeParams
from tests.research.synthetic import build_synthetic_warehouse, calendar

NOW = datetime(2026, 11, 11, 9, 0, tzinfo=UTC)
FACTORS = FactorParams(min_adv_value=0, min_session_index=0, min_stocks=1, min_industry=2)  # 3 synthetic stocks
REGIMES = RegimeParams(window=100, min_obs=50, breadth_min_adv=0)                          # 300 sessions

# Every table of research.duckdb. A new table must be added here, which forces the question.
POINT_IN_TIME = ("daily_panel", "market_daily", "source_coverage", "stock_features", "stock_factors",
                 "market_regimes", "stock_events", "stock_trading_costs")
OUTCOMES = ("stock_targets",)
METADATA = ("feature_builds", "factor_definitions")
VIEWS = ("feature_target",)                 # stock_features joined with stock_targets
# Back-adjusted price levels: valid only in ratios, and restated by the vendor after each corporate action.
ADJUSTED_LEVELS = ("adj_open", "adj_high", "adj_low", "adj_close", "mkt_open", "mkt_close", "mkt_ma50", "mkt_ma200")

# Every horizon of each outcome column (sessions after t that the column reads).
TARGET_HORIZONS = {"fwd_ret_close_1d": 1, "fwd_ret_close_3d": 3, "fwd_ret_close_5d": 5, "fwd_ret_close_10d": 10,
                   "fwd_ret_exec_3d": 3, "fwd_ret_exec_5d": 5, "fwd_ret_exec_10d": 10, "fwd_ret_exec_20d": 20,
                   "fwd_excess_exec_5d": 5, "fwd_excess_exec_10d": 10, "fwd_excess_exec_20d": 20,
                   "fwd_max_return_5d": 5, "fwd_max_drawdown_5d": 5, "y_up3_5d": 5}

# Period boundaries (CLAUDE.md, research integrity): last day of research, validation and holdout.
BOUNDARIES = (date(2023, 12, 31), date(2025, 12, 31), date(2026, 10, 2))


def _build(wh: Path, rs: Path) -> Path:
    build(wh, rs, NOW, FACTORS, REGIMES)
    return rs


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    d = tmp_path_factory.mktemp("integrity")
    build_synthetic_warehouse(d / "w.duckdb")
    return d / "w.duckdb", _build(d / "w.duckdb", d / "r.duckdb")


def _variant(base_wh: Path, tmp: Path, sql: str, params: list) -> Path:
    wh = tmp / "w.duckdb"
    wh.write_bytes(base_wh.read_bytes())
    with Warehouse(wh) as w:
        w.connection.execute(sql, params)
    return _build(wh, tmp / "r.duckdb")


def _table(path: Path, table: str, where: str = "TRUE", params: list | None = None,
           drop: tuple[str, ...] = ()) -> pd.DataFrame:
    with duckdb.connect(str(path), read_only=True) as con:
        df = con.execute(f"SELECT * FROM {table} WHERE {where} ORDER BY ALL", params or []).df()
    if "metadata" in df.columns:        # JSON text: floats are compared as numbers, not as their last printed digit
        meta = pd.json_normalize(df.pop("metadata").map(json.loads).tolist()).add_prefix("metadata.")
        df = pd.concat([df, meta], axis=1)
    return df.drop(columns=[c for c in drop if c in df.columns]).reset_index(drop=True)


def test_every_research_table_is_classified(base):
    with duckdb.connect(str(base[1]), read_only=True) as con:
        found = {(n, t) for n, t in con.execute(
            "SELECT table_name, table_type FROM information_schema.tables WHERE table_schema = 'main'").fetchall()}
    tables = {n for n, t in found if t == "BASE TABLE"}
    assert tables == set(POINT_IN_TIME + OUTCOMES + METADATA)
    assert {n for n, t in found if t == "VIEW"} == set(VIEWS)


def test_changing_the_future_changes_no_point_in_time_row(base, tmp_path):
    cut = calendar()[180]
    changed = _variant(base[0], tmp_path, """
        UPDATE quotes_daily SET price_open = price_open * 1.3, price_high = price_high * 1.4,
            price_low = price_low * 1.2, price_close = price_close * 1.3, total_value = total_value * 5,
            deal_volume = deal_volume * 3, buy_quantity = buy_quantity * 2, sell_count = sell_count + 7,
            buy_foreign_value = buy_foreign_value + 1e9
        WHERE date > ? AND price_close > 0""", [cut])
    for table in POINT_IN_TIME:
        pd.testing.assert_frame_equal(_table(base[1], table, "date <= ?", [cut]),
                                      _table(changed, table, "date <= ?", [cut]), obj=table)
    later = [t for t in POINT_IN_TIME if not _table(base[1], t, "date > ?", [cut]).equals(
        _table(changed, t, "date > ?", [cut]))]
    assert set(later) >= {"daily_panel", "market_daily", "stock_features", "stock_factors", "market_regimes"}
    assert not _table(base[1], "stock_targets", "date <= ?", [cut]).equals(
        _table(changed, "stock_targets", "date <= ?", [cut]))          # outcomes do look past the cutoff


def test_restated_adjustment_ratios_change_only_adjusted_levels(base, tmp_path):
    """A corporate action after the cutoff makes the vendor rescale adj_ratio on every earlier row."""
    cut = calendar()[200]
    changed = _variant(base[0], tmp_path, "UPDATE quotes_daily SET adj_ratio = adj_ratio * 1.25 WHERE date <= ?",
                       [cut])
    # outcomes are compared where their whole window (20 sessions at most) ends at or before the cutoff
    last = {t: cut for t in POINT_IN_TIME} | {t: calendar()[180] for t in OUTCOMES}
    for table, until in last.items():
        a = _table(base[1], table, "date <= ?", [until], drop=ADJUSTED_LEVELS)
        b = _table(changed, table, "date <= ?", [until], drop=ADJUSTED_LEVELS)
        pd.testing.assert_frame_equal(a, b, rtol=1e-9, atol=1e-12, obj=table)
    levels = _table(changed, "daily_panel", "date <= ?", [cut])["adj_close"]
    assert (levels < _table(base[1], "daily_panel", "date <= ?", [cut])["adj_close"]).all()


def test_one_symbol_never_moves_another_symbols_series(base, tmp_path):
    changed = _variant(base[0], tmp_path, """
        UPDATE quotes_daily SET price_open = price_open * 1.5, price_high = price_high * 1.5,
            price_low = price_low * 1.5, price_close = price_close * 1.5, price_basic = price_basic * 1.5,
            total_value = total_value * 3, deal_volume = deal_volume * 3, buy_quantity = buy_quantity * 2,
            buy_foreign_value = buy_foreign_value + 1e9
        WHERE symbol = 'UPC'""", [])
    for table in ("daily_panel", "stock_features", "stock_events", "stock_trading_costs") + OUTCOMES:
        pd.testing.assert_frame_equal(_table(base[1], table, "symbol <> 'UPC'"),
                                      _table(changed, table, "symbol <> 'UPC'"), obj=table)
    assert not _table(base[1], "stock_features", "symbol = 'UPC'").equals(
        _table(changed, "stock_features", "symbol = 'UPC'"))


def test_outcome_windows_past_the_last_session_are_null(base):
    t = _table(base[1], "stock_targets")
    for sym, g in t.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True)
        n = len(g)
        for col, h in TARGET_HORIZONS.items():
            assert g[col].iloc[n - h:].isna().all(), (sym, col)
        assert not g["target_complete"].iloc[n - 20:].any() and g["target_complete"].iloc[:n - 20].all(), sym
        for h in (1, 3, 5, 10, 20):
            assert g[f"crosses_period_{h}d"].iloc[n - h:].isna().all(), (sym, h)


def _sql_boundaries(text: str) -> list[tuple[date, date, date]]:
    """Every CASE that maps a date to a period, as its three boundary dates."""
    pattern = (r"<= DATE '(\d{4}-\d{2}-\d{2})' THEN 'research'\s+WHEN [\w.]+ <= DATE '(\d{4}-\d{2}-\d{2})' THEN "
               r"'validation'\s+WHEN [\w.]+ <= DATE '(\d{4}-\d{2}-\d{2})' THEN 'holdout'\s+ELSE 'forward'")
    return [tuple(date.fromisoformat(d) for d in m) for m in re.findall(pattern, text)]


def test_period_boundaries_agree_everywhere():
    from quant_research import daily, factor_report
    from quant_research.backtest.runner import PERIOD_RANGES
    from quant_research.cross import describe

    sql = resources.files("quant_research").joinpath("sql")
    found = {name: _sql_boundaries(sql.joinpath(name).read_text())
             for name in ("01_daily_panel.sql", "04_stock_targets.sql")}
    assert len(found["01_daily_panel.sql"]) == 1 and len(found["04_stock_targets.sql"]) == 6
    assert all(b == BOUNDARIES for blocks in found.values() for b in blocks)
    assert [PERIOD_RANGES[p][1] for p in ("research", "validation", "holdout")] == list(BOUNDARIES)
    starts = [PERIOD_RANGES[p][0] for p in ("validation", "holdout", "forward")]
    assert [(s - e).days for s, e in zip(starts, BOUNDARIES)] == [1, 1, 1]
    assert daily.FORWARD_START == PERIOD_RANGES["forward"][0]
    assert describe.RESEARCH_END == BOUNDARIES[0].isoformat()
    assert f"DATE '{BOUNDARIES[0].isoformat()}'" in factor_report.RESEARCH


def test_built_periods_follow_the_boundaries(tmp_path):
    wh = tmp_path / "w.duckdb"
    build_synthetic_warehouse(wh, start=date(2023, 11, 1))              # 300 sessions: research .. validation
    rs = _build(wh, tmp_path / "r.duckdb")
    p = _table(rs, "daily_panel")
    expect = p["date"].dt.date.map(lambda d: "research" if d <= BOUNDARIES[0] else
                                   "validation" if d <= BOUNDARIES[1] else
                                   "holdout" if d <= BOUNDARIES[2] else "forward")
    assert (p["period"] == expect).all()
    assert set(p["period"]) == {"research", "validation"}

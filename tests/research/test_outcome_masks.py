"""Point-in-time universe and outcome masks (review of 2026-11-10): the default sample is the recorded
definition, each mask is separate, and the per-horizon masks use the target's own window."""

from datetime import UTC, datetime

import duckdb
import pandas as pd
import pytest

from quant_research.build import build
from quant_research.results import OUTCOME_MASKS, ResearchParams, stored_params
from tests.research.synthetic import build_synthetic_warehouse

BUILT_AT = datetime(2027, 3, 1, 19, 0, tzinfo=UTC)
LOOSE = {"min_adv_value": 0, "min_session_index": 0}        # the synthetic market is small and illiquid


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("masks")
    build_synthetic_warehouse(tmp / "warehouse.duckdb")
    build(tmp / "warehouse.duckdb", tmp / "research.duckdb", BUILT_AT)
    c = duckdb.connect(str(tmp / "research.duckdb"), read_only=True)
    yield c
    c.close()


def keys(con, where: str) -> set:
    return set(con.execute(f"SELECT symbol, date FROM feature_target WHERE {where}").fetchall())


def test_default_sample_is_the_recorded_universe(con):
    p = ResearchParams(**LOOSE)
    assert keys(con, p.sample_sql()) == keys(con, p.universe_sql())
    assert keys(con, p.sample_sql("fwd_excess_exec_5d")) == keys(con, p.universe_sql())   # legacy: target ignored
    baseline = ResearchParams(**LOOSE, outcome_masks=())
    assert baseline.sample_sql() == baseline.point_in_time_sql() + " AND TRUE"
    assert keys(con, baseline.sample_sql()) > keys(con, p.sample_sql())                 # the masks do remove rows


def test_each_mask_removes_its_own_rows(con):
    pit = keys(con, ResearchParams(**LOOSE).point_in_time_sql())
    removed = {}
    for mask in OUTCOME_MASKS:
        p = ResearchParams(**LOOSE, outcome_masks=(mask,))
        removed[mask] = pit - keys(con, p.sample_sql())
    assert removed["execution"] == pit & keys(con, "coalesce(entry_blocked, false)")
    assert removed["period"] == pit & keys(con, "coalesce(crosses_period, false)")
    assert removed["quality"] == pit & keys(con, "coalesce(fwd_has_price_jump_20d, false)")
    assert removed["period"] and removed["execution"]                                  # non-vacuous here
    union = pit - keys(con, ResearchParams(**LOOSE).sample_sql())
    assert union == removed["quality"] | removed["execution"] | removed["period"]


def test_per_horizon_masks_use_the_target_window(con):
    panel = con.execute("SELECT symbol, date, period, price_jump FROM daily_panel ORDER BY symbol, date").df()
    t = con.execute("SELECT * FROM stock_targets").df().set_index(["symbol", "date"]).sort_index()
    period_end = {"research": "2023-12-31", "validation": "2025-12-31", "holdout": "2026-10-02",
                  "forward": "2099-12-31"}
    for h in (1, 3, 5, 10, 20):
        g = panel.groupby("symbol")
        ahead = pd.concat([g["price_jump"].shift(-k) for k in range(1, h + 1)], axis=1)
        jump = ahead.fillna(False).astype(bool).any(axis=1)
        d_h = g["date"].shift(-h)
        crosses = pd.Series([None if pd.isna(d) else pd.Timestamp(d) > pd.Timestamp(period_end[p])
                             for d, p in zip(d_h, panel["period"])], index=panel.index)
        exp = pd.DataFrame({"symbol": panel["symbol"], "date": panel["date"], "jump": jump, "crosses": crosses})
        exp = exp.set_index(["symbol", "date"]).sort_index()
        jump_col = "fwd_has_price_jump_20d" if h == 20 else f"fwd_has_price_jump_{h}d"
        assert (t[jump_col].astype(bool) == exp["jump"]).all(), h
        got, want = t[f"crosses_period_{h}d"], exp["crosses"]
        assert (got.isna() == want.isna()).all(), h
        assert (got[got.notna()].astype(bool) == want[want.notna()].astype(bool)).all(), h
    assert t["crosses_period_20d"].equals(t["crosses_period"])
    assert (t["crosses_period_5d"].fillna(False) <= t["crosses_period"].fillna(True)).all()   # nested windows
    assert t["crosses_period_5d"].fillna(False).sum() < t["crosses_period"].fillna(False).sum()


def test_horizon_window_sql_and_validation():
    p = ResearchParams(mask_window="horizon")
    assert "fwd_has_price_jump_5d" in p.mask_sql("fwd_excess_exec_5d")
    assert "crosses_period_5d" in p.mask_sql("fwd_excess_exec_5d")
    assert "crosses_period_20d" in p.mask_sql()                      # several horizons: the longest
    assert ResearchParams(outcome_masks=("period", "quality")).mask_sql() == \
        ResearchParams(outcome_masks=("quality", "period")).mask_sql()
    with pytest.raises(ValueError):
        p.mask_sql("fwd_max_return_7d")
    with pytest.raises(ValueError):
        ResearchParams(outcome_masks=("lookahead",))
    with pytest.raises(ValueError):
        ResearchParams(mask_window="h")


def test_stored_params_unchanged_for_the_recorded_definition():
    default = stored_params(ResearchParams())
    assert "outcome_masks" not in default and "mask_window" not in default
    assert set(default) == {"min_adv_value", "min_session_index", "min_stocks_per_date", "cost_round_trip",
                            "horizons", "extra"}
    custom = stored_params(ResearchParams(outcome_masks=("quality",)))
    assert custom["outcome_masks"] == ("quality",) and custom["mask_window"] == "legacy20"

"""Event study (doc 24): what usually follows each catalog event, on the research period.

Descriptive by design (docs/event-engine-plan.md): figures are stored in `event_study_stats` and NOT added
to the hypothesis log. An event that looks useful is promoted later through a normal, logged pattern test or
a pre-registration. The lift vs the universe and its Newey-West t are shown for orientation only.
"""

from dataclasses import asdict
from datetime import datetime

import numpy as np
import pandas as pd

from quant_research.events import EVENT_TYPES, catalog_version
from quant_research.provenance import from_runs
from quant_research.results import ResearchParams, ResultsStore
from quant_research.stats import date_means, mean_test

SCHEMA = """
CREATE TABLE IF NOT EXISTS event_study_stats (
    run_id VARCHAR NOT NULL, event_type VARCHAR NOT NULL, horizon VARCHAR NOT NULL, segment VARCHAR NOT NULL,
    n_events BIGINT, n_dates BIGINT, mean DOUBLE, median DOUBLE, win_rate DOUBLE, std DOUBLE, p05 DOUBLE,
    p95 DOUBLE, mean_max_gain_5d DOUBLE, mean_max_loss_5d DOUBLE, date_mean DOUBLE, lift DOUBLE, lift_t DOUBLE,
    PRIMARY KEY (run_id, event_type, horizon, segment));
"""
# label: (column, horizon in sessions, lift vs universe?)
HORIZONS = {
    "close_1d": ("fwd_ret_close_1d", 1, False),
    "close_3d": ("fwd_ret_close_3d", 3, False),
    "close_5d": ("fwd_ret_close_5d", 5, False),
    "exec_excess_5d": ("fwd_excess_exec_5d", 5, True),
    "exec_excess_10d": ("fwd_excess_exec_10d", 10, True),
    "exec_excess_20d": ("fwd_excess_exec_20d", 20, True),
}
SEGMENTS = ("all", "regime=Bull", "regime=Sideway", "regime=Bear", "exchange=HSX", "exchange=HNX", "exchange=UPCOM")


def _segment_mask(df: pd.DataFrame, seg: str) -> pd.Series:
    if seg == "all":
        return pd.Series(True, index=df.index)
    key, value = seg.split("=")
    return df["market_regime" if key == "regime" else "exchange_now"] == value


def stats_row(df: pd.DataFrame, col: str, h: int, lift_universe: tuple | None) -> dict:
    """Doc 24 figures for one event x horizon x segment (event level), plus the date-level lift."""
    y = df[col].to_numpy(dtype=float)
    ok = np.isfinite(y)
    y, sub = y[ok], df[ok]
    if len(y) == 0:
        return {}
    dates = sub["date"].to_numpy().astype("datetime64[D]").astype(np.int64)
    d, dm, _ = date_means(dates, y)
    lift = lift_t = None
    if lift_universe is not None:
        ud, um = lift_universe
        common, ia, ib = np.intersect1d(d, ud, return_indices=True)
        diff = dm[ia] - um[ib]
        if len(diff):
            lift = float(diff.mean())
            lift_t = mean_test(diff, h - 1)[1]
    gain, loss = sub["fwd_max_return_5d"].to_numpy(float), sub["fwd_max_drawdown_5d"].to_numpy(float)
    return {"n_events": int(len(y)), "n_dates": int(len(d)), "mean": float(y.mean()), "median": float(np.median(y)),
            "win_rate": float((y > 0).mean()), "std": float(y.std(ddof=1)) if len(y) > 1 else None,
            "p05": float(np.quantile(y, 0.05)), "p95": float(np.quantile(y, 0.95)),
            "mean_max_gain_5d": float(np.nanmean(gain)) if np.isfinite(gain).any() else None,
            "mean_max_loss_5d": float(np.nanmean(loss)) if np.isfinite(loss).any() else None,
            "date_mean": float(dm.mean()), "lift": lift, "lift_t": lift_t}


def run_event_study(store: ResultsStore, now: datetime, params: ResearchParams = ResearchParams(),
                    period: str = "research") -> str:
    if period != "research":
        raise ValueError("the event study is descriptive and runs on the research period only")
    version = catalog_version()
    params = ResearchParams(**{**asdict(params), "extra": {**params.extra, "event_catalog": version}})
    run_id = f"{now:%Y%m%dT%H%M%S}-event_study-catalog_{version}-{period}"
    store.con.execute(SCHEMA)
    store.start_run(run_id, "event_study", period, params, now, ("event_catalog", version))
    cols = ", ".join(c for c, _, _ in HORIZONS.values())
    df = store.con.execute(f"""
        SELECT e.event_type, f.symbol, f.date, f.market_regime, f.exchange_now, {cols},
               f.fwd_max_return_5d, f.fwd_max_drawdown_5d
        FROM rs.stock_events e JOIN rs.feature_target f USING (symbol, date)
        WHERE {params.sample_sql()} AND f.period = ?""", [period]).df()
    df["date"] = pd.to_datetime(df["date"])
    universe = {}
    for label, (col, _h, with_lift) in HORIZONS.items():
        if with_lift:
            ud, um, _ = store.universe_by_date(period, params, col)
            universe[label] = (ud, um)
    rows = []
    for event in EVENT_TYPES:
        ev = df[df["event_type"] == event]
        for seg in SEGMENTS:
            part = ev[_segment_mask(ev, seg)]
            for label, (col, h, _) in HORIZONS.items():
                r = stats_row(part, col, h, universe.get(label) if seg == "all" else None)
                if r:
                    rows.append({"run_id": run_id, "event_type": event, "horizon": label, "segment": seg, **r})
    if rows:
        frame = pd.DataFrame(rows)
        store.con.register("frame", frame)
        store.con.execute(f"INSERT INTO event_study_stats SELECT {', '.join(frame.columns)} FROM frame")
        store.con.unregister("frame")
    store.finish_run(run_id)
    return run_id


def render_event_study(store: ResultsStore, run_id: str, cost: float) -> str:
    s = store.con.execute("""SELECT event_type, horizon, n_events, n_dates, mean, median, win_rate, std, p05, p95,
                                    mean_max_gain_5d, mean_max_loss_5d, lift, lift_t
                             FROM event_study_stats WHERE run_id = ? AND segment = 'all'
                             ORDER BY event_type, horizon""", [run_id]).df()
    lines = [f"# Event study - run `{run_id}`", "", *from_runs(store.con, [run_id]),
             "- Research period, liquid universe (same filter as Phase 3). Descriptive: **not** in the hypothesis "
             "log; the lift t-statistics are not corrected for multiple testing.",
             f"- close_* = raw future close-to-close returns (doc 24); exec_excess_* = enter at the next open, "
             f"minus VNINDEX (tradable view). Round-trip cost {cost:.1%} not subtracted.",
             "- mean_max_gain_5d / mean_max_loss_5d = average best / worst point within 5 sessions after the "
             "next open.", ""]
    for event, desc in EVENT_TYPES.items():
        part = s[s["event_type"] == event].drop(columns=["event_type"])
        if part.empty:
            continue
        lines += [f"## {event}", "", f"Condition: {desc}", "", _md(part)]
    return "\n".join(lines) + "\n"


def _md(df: pd.DataFrame) -> str:
    out = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(out) + "\n"

"""Pattern tests and feature scans on `rs.feature_target` (research database attached read-only)."""

from dataclasses import asdict
from datetime import datetime

import numpy as np
from scipy import stats as sps

from quant_research.patterns import LIBRARY, Pattern
from quant_research.results import ResearchParams, ResultsStore, check_period
from quant_research.stats import date_means, mean_test, summarize

DESCRIPTIVE = {"close_1d": "fwd_ret_close_1d", "close_3d": "fwd_ret_close_3d"}
REGIMES = ("Bull", "Sideway", "Bear")
EXCHANGES = ("HSX", "HNX", "UPCOM")
SCAN_FEATURES = (
    "return_1d", "return_3d", "return_5d", "return_10d", "return_20d", "excess_return_5d", "excess_return_20d",
    "volume_ratio_5", "volume_ratio_20", "volume_change", "volume_zscore_20", "volatility_5", "volatility_20",
    "atr_14_pct", "high_low_range", "gap", "close_position", "close_vs_avg_price", "order_imbalance",
    "volume_imbalance", "buy_pressure", "sell_pressure", "buy_sell_ratio", "foreign_net_volume",
    "foreign_net_value", "foreign_net_3d", "foreign_net_5d", "foreign_net_20d", "foreign_net_ratio_20",
    "foreign_intensity",
)


def outcome_column(horizon: int) -> str:
    return f"fwd_excess_exec_{horizon}d"


def _run_id(kind: str, name: str, period: str, now: datetime) -> str:
    return f"{now:%Y%m%dT%H%M%S}-{kind}-{name}-{period}"


def _events(store: ResultsStore, source_sql: str, bind: list, params: ResearchParams) -> dict[str, np.ndarray]:
    cols = ["symbol", "date", "market_regime", "exchange_now"] + [outcome_column(h) for h in params.horizons] \
        + list(DESCRIPTIVE.values())
    rows = store.con.execute(f"SELECT {', '.join(cols)} FROM ({source_sql})", bind).fetchnumpy()
    # fetchnumpy returns masked arrays where values are NULL: fill explicitly. np.asarray() would expose
    # the arbitrary data under the mask (e.g. 0.0 instead of a missing outcome).
    out: dict[str, np.ndarray] = {}
    for key, values in rows.items():
        if key == "date":
            out[key] = np.asarray(values).astype("datetime64[D]").astype(np.int64)
        elif key in ("symbol", "market_regime", "exchange_now"):
            out[key] = np.ma.filled(np.ma.asarray(values, dtype=object), None)
        else:
            out[key] = np.ma.filled(np.ma.asarray(values, dtype=float), np.nan)
    return out


def _definition_text(pattern: Pattern) -> str:
    if not pattern.decile_of:
        return pattern.where
    return f"{pattern.where}  [cross-sectional decile {pattern.decile} of {pattern.decile_of}, per date]"


def run_pattern(store: ResultsStore, pattern: Pattern, period: str, params: ResearchParams, now: datetime,
                final: bool = False, library: dict[str, Pattern] = LIBRARY, table: str = "rs.feature_target") -> str:
    """`table`: feature_target or a view extending it with more columns (e.g. cross-stock features)."""
    check_period(period, final)
    run_id = _run_id("pattern", pattern.name, period, now)
    store.con.execute("INSERT OR IGNORE INTO pattern_definitions VALUES (?, ?, ?, ?, ?, ?, ?)",
                      [pattern.name, pattern.version, _definition_text(pattern), pattern.base, pattern.hypothesis,
                       pattern.doc_ref, now])
    store.start_run(run_id, "pattern", period, params, now, (pattern.name, pattern.version))
    source = pattern.source_sql(params.sample_sql(), params.min_stocks_per_date, table=table)
    ev = _events(store, source, [period], params)

    segments = {"all": np.ones(len(ev["date"]), dtype=bool)}
    segments |= {f"regime={r}": ev["market_regime"] == r for r in REGIMES}
    segments |= {f"exchange={e}": ev["exchange_now"] == e for e in EXCHANGES}
    horizons = {f"exec_excess_{h}d": (outcome_column(h), h) for h in params.horizons}
    horizons |= {label: (col, int(label.split("_")[1][:-1])) for label, col in DESCRIPTIVE.items()}

    stat_rows = []
    for label, (col, h) in horizons.items():
        for seg, mask in segments.items():
            s = summarize(ev["date"][mask], ev[col][mask], h, params.cost_round_trip)
            stat_rows.append([run_id, label, seg, *asdict(s).values()])
    store.con.executemany(f"INSERT INTO pattern_stats VALUES ({', '.join(['?'] * 16)})", stat_rows)
    for h in params.horizons:
        _lift_vs_universe(store, run_id, ev, h, period, params, now)

    store.con.execute(f"""
        INSERT INTO pattern_occurrences
        SELECT ?, symbol, date, market_regime, exchange_now,
               fwd_excess_exec_5d, fwd_excess_exec_10d, fwd_excess_exec_20d
        FROM ({source})""", [run_id, period])

    if pattern.base:
        _compare_with_base(store, run_id, pattern, library[pattern.base], ev, period, params, now, table)
    store.finish_run(run_id)
    return run_id


def _lift_vs_universe(store: ResultsStore, run_id: str, ev: dict, h: int, period: str,
                      params: ResearchParams, now: datetime) -> None:
    """Primary test (logged): pattern date-mean minus universe date-mean on the same dates."""
    col = outcome_column(h)
    mask = np.isfinite(ev[col])
    pd_, pm, _ = date_means(ev["date"][mask], ev[col][mask])
    ud, um, uwin = store.universe_by_date(period, params, col)
    common, ia, ib = np.intersect1d(pd_, ud, return_indices=True)
    lift = pm[ia] - um[ib]
    se, t, p, _, _ = mean_test(lift, h - 1)
    label = f"exec_excess_{h}d"
    store.con.execute("INSERT INTO pattern_lifts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      [run_id, label, len(common), float(pm[ia].mean()) if len(common) else None,
                       float(um[ib].mean()) if len(common) else None, uwin,
                       float(lift.mean()) if len(lift) else None, se, t, p])
    store.log_hypothesis(run_id, f"lift_{label}", period, p, now)


def _compare_with_base(store: ResultsStore, run_id: str, pattern: Pattern, base: Pattern, ev: dict, period: str,
                       params: ResearchParams, now: datetime, table: str = "rs.feature_target") -> None:
    """Refined events vs base events that do NOT meet the refinement, date by date (doc 5.2)."""
    base_src = base.source_sql(params.sample_sql(), params.min_stocks_per_date, table=table)
    refined_src = pattern.source_sql(params.sample_sql(), params.min_stocks_per_date, table=table)
    # base events that do not meet the refinement; each source binds the period once
    rest = _events(store, f"SELECT b.* FROM ({base_src}) b ANTI JOIN ({refined_src}) r USING (symbol, date)",
                   [period, period], params)
    for h in params.horizons:
        col = outcome_column(h)
        a_mask, b_mask = np.isfinite(ev[col]), np.isfinite(rest[col])
        da, ma, _ = date_means(ev["date"][a_mask], ev[col][a_mask])
        db, mb, _ = date_means(rest["date"][b_mask], rest[col][b_mask])
        common, ia, ib = np.intersect1d(da, db, return_indices=True)
        diff = ma[ia] - mb[ib]
        se, t, p, _, _ = mean_test(diff, h - 1)
        store.con.execute("INSERT INTO pattern_comparisons VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                          [run_id, f"exec_excess_{h}d", base.name, len(common),
                           float(diff.mean()) if len(diff) else None, se, t, p])
        store.log_hypothesis(run_id, f"vs_base_exec_excess_{h}d", period, p, now)


def run_scan(store: ResultsStore, period: str, params: ResearchParams, now: datetime, final: bool = False,
             features: tuple[str, ...] = SCAN_FEATURES, table: str = "rs.feature_target", row_filter: str = "TRUE",
             name: str = "features", kind: str = "scan") -> str:
    """Cross-sectional decile scan per date for every feature x horizon (doc 26, Phase 3).

    `table` / `row_filter` (fixed SQL, never user input) restrict the scan, e.g. to followers of industry
    leaders in the cross-stock view.
    """
    check_period(period, final)
    run_id = _run_id(kind, name, period, now)
    store.start_run(run_id, kind, period, params, now)
    con = store.con
    con.execute(f"""CREATE OR REPLACE TEMP TABLE scan_universe AS
        SELECT * FROM {table} WHERE {params.sample_sql()} AND ({row_filter}) AND period = ?""", [period])
    segments = {"all": "TRUE"} | {f"regime={r}": f"market_regime = '{r}'" for r in REGIMES} \
        | {f"exchange={e}": f"exchange_now = '{e}'" for e in EXCHANGES}

    for feature in features:
        for h in params.horizons:
            y = outcome_column(h)
            for seg, cond in segments.items():
                part = "date, exchange_now" if seg.startswith("exchange=") else "date"
                base = f"""SELECT {part}, symbol, {feature} AS f, {y} AS y FROM scan_universe
                           WHERE {cond} AND {feature} IS NOT NULL AND isfinite({feature}) AND {y} IS NOT NULL"""
                dec = con.execute(f"""
                    WITH b AS ({base}),
                    d AS (SELECT *, ntile(10) OVER (PARTITION BY {part} ORDER BY f, symbol) AS decile,   -- ties: by symbol
                                 count(*) OVER (PARTITION BY {part}) AS n_day FROM b)
                    SELECT date, decile, avg(y) AS m, count(*) AS n FROM d
                    WHERE n_day >= {params.min_stocks_per_date} GROUP BY date, decile ORDER BY date, decile
                """).fetchnumpy()
                ic = con.execute(f"""
                    WITH b AS ({base}),
                    r AS (SELECT date,                       -- average ranks: exact Spearman with ties
                                 rank() OVER (PARTITION BY {part} ORDER BY f)
                                     + (count(*) OVER (PARTITION BY {part}, f) - 1) / 2.0 AS rf,
                                 rank() OVER (PARTITION BY {part} ORDER BY y)
                                     + (count(*) OVER (PARTITION BY {part}, y) - 1) / 2.0 AS ry,
                                 count(*) OVER (PARTITION BY {part}) AS n_day FROM b)
                    SELECT date, corr(rf, ry) AS ic FROM r WHERE n_day >= {params.min_stocks_per_date}
                    GROUP BY date HAVING corr(rf, ry) IS NOT NULL AND isfinite(corr(rf, ry)) ORDER BY date
                """).fetchnumpy()
                _store_scan(store, run_id, feature, f"exec_excess_{h}d", seg, h, dec, ic, period, now)
    store.finish_run(run_id)
    return run_id


def _store_scan(store: ResultsStore, run_id: str, feature: str, horizon: str, seg: str, h: int,
                dec: dict, ic: dict, period: str, now: datetime) -> None:
    # Safe to use asarray here: these columns can never be NULL or NaN (filtered in SQL).
    dates, deciles, means, counts = (np.asarray(dec[k]) for k in ("date", "decile", "m", "n"))
    rows, decile_means = [], {}
    for d in range(1, 11):
        mask = deciles == d
        if mask.any():
            _, per_date, _ = date_means(dates[mask].astype("datetime64[D]").astype(np.int64), means[mask].astype(float))
            decile_means[d] = float(per_date.mean())
            rows.append([run_id, feature, horizon, seg, d, decile_means[d], int(mask.sum()), int(counts[mask].sum())])
    if rows:
        store.con.executemany("INSERT INTO scan_deciles VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)

    top = {dt: m for dt, dd, m in zip(dates, deciles, means) if dd == 10}
    bottom = {dt: m for dt, dd, m in zip(dates, deciles, means) if dd == 1}
    both = sorted(set(top) & set(bottom))
    spread = np.array([top[d] - bottom[d] for d in both], dtype=float)
    _, s_t, s_p, _, _ = mean_test(spread, h - 1)
    ic_series = np.asarray(ic["ic"], dtype=float)
    ic_series = ic_series[np.isfinite(ic_series)]   # corr() is NaN (not NULL) on zero-variance days
    _, i_t, i_p, _, _ = mean_test(ic_series, h - 1)
    mono = None
    if len(decile_means) == 10:
        mono = float(sps.spearmanr(np.arange(1, 11), [decile_means[d] for d in range(1, 11)]).statistic)
    store.con.execute("INSERT INTO scan_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                      [run_id, feature, horizon, seg, len(spread), float(spread.mean()) if len(spread) else None,
                       s_t, s_p, float(ic_series.mean()) if len(ic_series) else None, i_t, i_p, mono])
    if seg == "all":
        store.log_hypothesis(run_id, f"spread_{feature}_{horizon}", period, s_p, now)

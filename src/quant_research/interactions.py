"""Interaction Engine (docs/regime-interaction-plan.md, part 2): is a factor more informative in regime A than B?

Daily IC = Spearman correlation between a factor and the 10-session execution excess return, per date over
the scored stocks that pass the research universe. Each factor x dimension test regresses the daily IC on a
dummy (state A = 1, state B = 0, other dates dropped) with Newey-West errors (lag 9). Research period only:
validation and holdout are used, forward needs a pre-registration.
"""

from datetime import date, datetime

import numpy as np
import pandas as pd
import scipy.stats as sps

from quant_research.cross.hypotheses import hac_ols
from quant_research.factors import FACTORS, version
from quant_research.results import ResearchParams, ResultsStore
from quant_research.stats import mean_test

HORIZON = 10
TARGET = f"fwd_excess_exec_{HORIZON}d"
CONTRASTS = {   # dimension: (state A, state B)
    "direction": ("Bull", "Bear"),
    "volatility": ("high", "low"),
    "liquidity": ("high", "low"),
    "breadth": ("strong", "weak"),
    "foreign": ("buying", "selling"),
    "risk": ("risk_on", "risk_off"),
}
HALF_SPLIT = date(2015, 1, 1)
MIN_DATES = 50            # per state; fewer gives no test (NULL p, excluded from q-values)
MIN_STOCKS = 30           # per date for an IC
RUN_NAME = "interaction_scan"


class InteractionRefused(ValueError):
    pass


def daily_ic(store: ResultsStore, period: str, params: ResearchParams) -> pd.DataFrame:
    """date, factor, ic, n: Spearman via Pearson on average ranks, recomputed within the eligible rows."""
    return store.con.execute(f"""
        WITH t AS (SELECT symbol, date, {TARGET} AS y FROM rs.feature_target
                   WHERE {params.sample_sql(TARGET)} AND period = ? AND {TARGET} IS NOT NULL),
        s AS (SELECT f.date, f.factor, f.value, t.y FROM rs.stock_factors f JOIN t USING (symbol, date)),
        r AS (SELECT date, factor,
                     rank() OVER (PARTITION BY date, factor ORDER BY value)
                       + (count(*) OVER (PARTITION BY date, factor, value) - 1) / 2.0 AS rx,
                     rank() OVER (PARTITION BY date, factor ORDER BY y)
                       + (count(*) OVER (PARTITION BY date, factor, y) - 1) / 2.0 AS ry
              FROM s)
        SELECT date, factor, corr(rx, ry) AS ic, count(*) AS n FROM r
        GROUP BY ALL HAVING count(*) >= ? ORDER BY date, factor""", [period, MIN_STOCKS]).df()


def contrast(ic: pd.Series, in_a: pd.Series, in_b: pd.Series) -> dict:
    """Difference of mean IC (A - B) with a HAC t; ic, in_a, in_b are aligned on date (sorted)."""
    keep = (in_a | in_b) & ic.notna() & np.isfinite(ic)
    y, x = ic[keep].to_numpy(float), in_a[keep].to_numpy(float)
    n_a, n_b = int(x.sum()), int(len(x) - x.sum())
    out = {"n_a": n_a, "n_b": n_b, "diff": None, "se": None, "t": None, "p": None}
    if n_a < MIN_DATES or n_b < MIN_DATES:
        return out
    beta, ses = hac_ols(y, x, HORIZON - 1)
    t = float(beta[1] / ses[1]) if ses[1] else None
    out.update(diff=float(beta[1]), se=float(ses[1]), t=t,
               p=None if t is None else float(2 * sps.t.sf(abs(t), len(y) - 2)))
    return out


def already_run(store: ResultsStore, period: str) -> str | None:
    row = store.con.execute("""SELECT run_id FROM research_runs WHERE status = 'ok' AND kind = 'interaction'
                               AND period = ? ORDER BY created_at DESC LIMIT 1""", [period]).fetchone()
    return row[0] if row else None


SCHEMA = """
CREATE TABLE IF NOT EXISTS interaction_stats (
    run_id VARCHAR NOT NULL, factor VARCHAR NOT NULL, dimension VARCHAR NOT NULL, state VARCHAR NOT NULL,
    n_dates INTEGER, mean_ic DOUBLE, t DOUBLE, PRIMARY KEY (run_id, factor, dimension, state));
CREATE TABLE IF NOT EXISTS interaction_tests (
    run_id VARCHAR NOT NULL, factor VARCHAR NOT NULL, dimension VARCHAR NOT NULL, state_a VARCHAR, state_b VARCHAR,
    n_a INTEGER, n_b INTEGER, diff DOUBLE, se DOUBLE, t DOUBLE, p DOUBLE, diff_first_half DOUBLE,
    diff_second_half DOUBLE, PRIMARY KEY (run_id, factor, dimension));
"""


def run_scan(store: ResultsStore, period: str, now: datetime, params: ResearchParams = ResearchParams()) -> str:
    if period != "research":
        raise InteractionRefused(f"The interaction scan runs on the research period only, not {period!r}: "
                                 "validation and holdout are used; forward needs a pre-registration.")
    earlier = already_run(store, period)
    if earlier:
        raise InteractionRefused(f"Run {earlier} already logged these tests; invalidate it with a reason first.")
    store.con.execute(SCHEMA)
    run_id = f"{now:%Y%m%dT%H%M%S}-interaction-{RUN_NAME}-{period}"
    store.start_run(run_id, "interaction", period, params, now, (RUN_NAME, version()))
    ic = daily_ic(store, period, params)
    cols = ", ".join(f'"{d}"' for d in CONTRASTS)                  # "foreign" is a SQL keyword
    regimes = store.con.execute(f"SELECT date, {cols} FROM rs.market_regimes").df().set_index("date")
    for factor in FACTORS:
        series = ic[ic["factor"] == factor].set_index("date")["ic"].sort_index()
        series = series[np.isfinite(series)]
        labels = regimes.reindex(series.index)
        _stat(store, run_id, factor, "all", "all", series)
        for dim, (a, b) in CONTRASTS.items():
            for state in sorted(labels[dim].dropna().unique()):
                _stat(store, run_id, factor, dim, state, series[labels[dim] == state])
            res = contrast(series, labels[dim] == a, labels[dim] == b)
            first = series.index < pd.Timestamp(HALF_SPLIT)
            halves = [_diff(series[m], labels[dim][m], a, b) for m in (first, ~first)]
            store.con.execute("INSERT INTO interaction_tests VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                              [run_id, factor, dim, a, b, res["n_a"], res["n_b"], res["diff"], res["se"], res["t"],
                               res["p"], *halves])
            store.log_hypothesis(run_id, f"ix_{factor}_{dim}", period, res["p"], now)
    store.finish_run(run_id)
    return run_id


def _stat(store: ResultsStore, run_id: str, factor: str, dim: str, state: str, s: pd.Series) -> None:
    _, t, _, _, _ = mean_test(s.to_numpy(float), HORIZON - 1)
    store.con.execute("INSERT INTO interaction_stats VALUES (?, ?, ?, ?, ?, ?, ?)",
                      [run_id, factor, dim, state, len(s), float(s.mean()) if len(s) else None, t])


def _diff(s: pd.Series, labels: pd.Series, a: str, b: str) -> float | None:
    sa, sb = s[labels == a], s[labels == b]
    return float(sa.mean() - sb.mean()) if len(sa) and len(sb) else None

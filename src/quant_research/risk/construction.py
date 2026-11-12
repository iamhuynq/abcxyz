"""Construction-grid configurations for the Risk Engine (docs/risk-engine-pr2-plan.md, decision D1).

The historical grid runs cannot be reproduced: their warehouse file is not kept and their code was uncommitted
(Phase 0 audit). So a construction portfolio is a **current re-simulation** of the declared configuration, with
the configuration stored by the latest valid run (its version must match), at the decision setting (1 bn VND,
cost model v1), with today's code and data. It is never presented as the historical run: the target is
`construction:<name>@resim`, and a reconciliation table compares it with that run's stored metrics (relative
tolerance 1e-9, the reproducibility policy H-1). Nothing is written to the grid's tables.
"""

import json
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from quant_research.backtest.data import UniverseRule
from quant_research.backtest.benchmarks import liquidity_weighted_curve
from quant_research.backtest.metrics import WRITEDOWN_SESSIONS, curve_metrics, equal_weight_curve, participation
from quant_research.portfolio import evaluate as grid
from quant_research.portfolio.construction import Construction
from quant_research.portfolio.engine import rebalance_days
from quant_research.results import ResultsStore
from quant_research.risk.data import Snapshot

TOLERANCE = 1e-9
METHOD = "current re-simulation (today's code and data, decision setting 1 bn VND, cost model v1 k = 1)"


@dataclass(frozen=True)
class Source:
    run_id: str
    name: str
    config: grid.Config
    rule: UniverseRule
    version: str


def source(store: ResultsStore, name: str) -> Source:
    """The latest valid run of the configuration, and the configuration and universe it stored."""
    row = store.con.execute("""SELECT run_id, pattern_version, params FROM research_runs WHERE kind = 'portfolio'
                               AND status = 'ok' AND pattern_name = ? ORDER BY created_at DESC LIMIT 1""",
                            [name]).fetchone()
    if row is None:
        raise ValueError(f"no valid construction run of configuration {name!r}")
    run_id, version, params = row
    extra = json.loads(params)["extra"]
    c = extra["config"]
    cfg = grid.Config(c["signal"], c["schedule"], Construction(**c["construction"]),
                      tuple(c["cash_when"]) if c.get("cash_when") else None)
    if grid.config_version(cfg) != version:
        raise ValueError(f"run {run_id} stored configuration version {version}, but its stored parameters give "
                         f"{grid.config_version(cfg)}: refusing to re-simulate something else")
    return Source(run_id, name, cfg, UniverseRule(**extra["universe"]), version)


def _positions(market, state: dict) -> pd.DataFrame:
    """The engine's own holdings and marks at the snapshot close (PortfolioResult.state): no valuation rule is
    re-derived here. A position whose mark is missing keeps a NaN value, which the assessment reports as unvalued."""
    shares, marks = state["shares"], state["marks"]
    held = np.flatnonzero(shares > 0)
    return pd.DataFrame([(str(market.symbols[j]), float(shares[j]), float(marks[j]), float(shares[j] * marks[j]))
                         for j in held], columns=["symbol", "shares", "price", "value"])


# Reconciliation contract. REQUIRED: every stored strategy metric of the decision setting that this re-simulation
# recomputes; all must be present on both sides and within TOLERANCE for "reconciles". NOT_RECOMPUTED: stored, but
# declared out of the comparison, with the reason. NULL_EQUIVALENT: metrics where NULL on both sides is a match.
SIMULATION_METRICS = ("cagr", "total_return", "volatility", "sharpe", "max_drawdown", "max_drawdown_sessions",
                      "turnover_per_year", "costs_paid_x_initial", "avg_names", "avg_cash_share",
                      "max_target_industry_weight", "max_actual_industry_weight", "industry_cap_breach_sessions",
                      "industry_cap_breach_after_trades", "orders_capped", "orders_blocked", "orders_blocked_no_adv",
                      "orders_blocked_t2", "orders_partial_t2", "stale_value_share_mean", "stale_value_share_max",
                      "cost_fallbacks", "data_error_exits", "participation_median", "participation_p95")
REQUIRED = SIMULATION_METRICS + ("rank_persistence", "excess_cagr_vs_equal_weight",
                                 "excess_cagr_vs_liquidity_weighted", "excess_cagr_vs_vnindex")
NOT_RECOMPUTED = {"break_even_extra_cost_per_side": "not recomputed: a 16-simulation bisection search"}
NULL_EQUIVALENT = {"participation_median": "both undefined: no order with a valid ADV",
                   "participation_p95": "both undefined: no order with a valid ADV"}


def _recomputed(prep, src: Source, res) -> dict[str, float]:
    m = prep.market
    out = grid._metrics(res, grid.DECISION[0])
    out |= participation(np.array([abs(t[2]) * t[3] for t in res.trades]),
                         np.array([m.adv_value[t[0] - 1, t[1]] for t in res.trades]))
    out["rank_persistence"] = grid.rank_persistence(prep.scores, rebalance_days(m.dates, src.config.schedule))
    cap = grid.DECISION[0]
    bench = {"equal_weight": equal_weight_curve(m.close, m.universe, cap),
             "liquidity_weighted": liquidity_weighted_curve(m.close, m.universe, m.adv_value, cap),
             "vnindex": cap * m.index_close / m.index_close[0]}
    out |= {f"excess_cagr_vs_{b}": out["cagr"] - curve_metrics(c)["cagr"] for b, c in bench.items()}
    return out


def _finite(x) -> bool:
    return x is not None and np.isfinite(x)


def reconciliation(store: ResultsStore, src: Source, res, prep) -> list[tuple]:
    """(metric, stored, re-simulated, relative difference, within, note) for every required metric (a missing one is
    a mismatch), every declared not-recomputed metric and any other stored metric (within None: not compared)."""
    now = _recomputed(prep, src, res)
    stored = dict(store.con.execute("""SELECT metric, value FROM portfolio_results WHERE run_id = ? AND capital = ?
                                       AND cost_model = ? AND series = 'strategy'""",
                                    [src.run_id, grid.DECISION[0], grid.DECISION[1]]).fetchall())
    out = []
    for metric in REQUIRED:
        a, b = stored.get(metric), now.get(metric)
        if metric not in stored:
            out.append((metric, None, b, None, False, "missing from the stored run"))
        elif metric not in now:
            out.append((metric, a, None, None, False, "not produced by the re-simulation"))
        elif not _finite(a) or not _finite(b):
            both = not _finite(a) and not _finite(b) and metric in NULL_EQUIVALENT
            out.append((metric, a, b, None, bool(both), NULL_EQUIVALENT[metric] if both else "undefined on one side or "
                        "both, and NULL is not a match for this metric"))
        else:
            diff = float(abs(b - a) / abs(a) if a else abs(b - a))
            out.append((metric, float(a), float(b), diff, bool(diff <= TOLERANCE), ""))
    for metric in sorted(set(stored) - set(REQUIRED)):
        out.append((metric, stored[metric], None, None, None,
                    NOT_RECOMPUTED.get(metric, "not compared: not a metric of the re-simulation")))
    return out


def reconciles(rows: list[tuple]) -> bool:
    """True only if every required metric was compared and is within tolerance."""
    req = [r for r in rows if r[0] in REQUIRED]
    return len(req) == len(REQUIRED) and all(r[4] is True for r in req)


def snapshot(store: ResultsStore, name: str, on: date | None = None) -> tuple[Snapshot, Source, list[tuple]]:
    src = source(store, name)
    prep = grid.prepare(store, src.config, "research", src.rule)
    dates = prep.market.dates.astype("datetime64[D]").astype(object)
    i = len(dates) - 1 if on is None else int(np.searchsorted(dates, on, side="right")) - 1
    if i < 0:
        raise ValueError(f"{on} is before the first session of configuration {name}")
    res = grid.simulate(prep, src.config, snapshot_at=i)
    written = grid.simulate(prep, src.config, writedown_after=WRITEDOWN_SESSIONS)
    state = res.state
    pos = _positions(prep.market, state)
    nav, cash = state["equity"], state["cash"]
    gap = nav - cash - float(pos["value"].sum())
    gap = 0.0 if abs(gap) < 0.5 else gap
    rec = reconciliation(store, src, res, prep)
    bad = [m for m, *_rest, w, _n in rec if w is False]
    skipped = [m for m, *_rest, w, _n in rec if w is None]
    verdict = ("reconciles with it" if reconciles(rec) else
               f"does not reconcile with it: {', '.join(bad)} differ beyond {TOLERANCE:g} or are missing")
    notes = [f"Re-simulation of {name} (version {src.version}) with today's code and data; NOT the historical run "
             f"{src.run_id}. It {verdict}."
             + (f" Not compared: {', '.join(skipped)}." if skipped else ""),
             "Positions, marks and cash are the engine's own state at the close; a second curve values positions "
             f"without a trade for more than {WRITEDOWN_SESSIONS} sessions at 0."]
    curves = {"default marks (last traded close)": (dates[:i + 1], res.equity[:i + 1]),
              f"write-down {WRITEDOWN_SESSIONS}": (dates[:i + 1], written.equity[:i + 1])}
    snap = Snapshot(f"construction:{name}@resim", dates[i], nav, cash, pos, curves, notes, gap)
    return snap, src, rec

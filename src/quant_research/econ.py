"""Economic evaluation (docs/economic-validation-plan.md): one strategy under cost models, capital levels,
benchmarks and matched controls. Research period only; descriptive (nothing goes to the hypothesis log).
"""

import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime

import numpy as np

from quant_research.backtest import runner
from quant_research.backtest.benchmarks import key_matrices, liquidity_weighted_curve, matched_random_selector
from quant_research.backtest.costs import CostModel
from quant_research.backtest.data import Market, UniverseRule
from quant_research.backtest.engine import Strategy, random_selector, run, signal_selector
from quant_research.backtest.metrics import (WRITEDOWN_SESSIONS, curve_metrics, equal_weight_curve, participation,
                                             strategy_metrics)
from quant_research import controls
from quant_research.results import ResearchParams, ResultsStore

CAPITALS = (1e9, 10e9, 100e9)
COSTS: dict[str, CostModel | None] = {"flat": None, "v1_k0.5": CostModel(k=0.5), "v1_k1": CostModel(k=1.0),
                                      "v1_k2": CostModel(k=2.0),
                                      "v1_k1_tick": CostModel(k=1.0, spread="tick")}   # spread lower bound
CONTROL_COSTS = ("flat", "v1_k1")
FLAT_AS_MODEL = CostModel(k=0.0, spread="none")      # flat costs, so extra_flat can be added on top
N_CONTROL = controls.N_CONTROL        # 20 before the review of 2026-11-10; recorded runs store theirs
BREAK_EVEN_CAPITAL = 1e9
BREAK_EVEN_MAX = 0.05          # per side

SCHEMA = """
CREATE TABLE IF NOT EXISTS economic_results (
    run_id VARCHAR NOT NULL, capital DOUBLE NOT NULL, cost_model VARCHAR NOT NULL, series VARCHAR NOT NULL,
    metric VARCHAR NOT NULL, value DOUBLE, PRIMARY KEY (run_id, capital, cost_model, series, metric));
"""


class EconRefused(ValueError):
    pass


def strategy_version(strategy: Strategy, rule: UniverseRule) -> str:
    """Same hash as backtest runs (runner.run_backtest), so the evaluated strategy is identifiable."""
    return hashlib.sha256(json.dumps([asdict(strategy), asdict(rule)], sort_keys=True).encode()).hexdigest()[:8]


def evaluate(store: ResultsStore, label: str, strategy: Strategy, period: str, now: datetime,
             rule: UniverseRule = UniverseRule(), n_control: int = N_CONTROL,
             capitals: tuple[float, ...] = CAPITALS) -> str:
    if period != "research":
        raise EconRefused(f"Economic evaluation runs on the research period only, not {period!r}: validation and "
                          "holdout are used; forward needs a pre-registration.")
    store.con.execute(SCHEMA)
    store.con.execute(controls.SCHEMA)
    full = runner._market(store, period, rule)
    market = runner._slice(full, runner._active_window(full))
    keys = key_matrices(store.con, market)
    run_id = f"{now:%Y%m%dT%H%M%S}-econ-{label}-{period}"
    params = ResearchParams(extra={"strategy": asdict(strategy), "universe": asdict(rule), "capitals": capitals,
                                   "costs": {k: None if v is None else asdict(v) for k, v in COSTS.items()},
                                   "n_control": n_control})
    store.start_run(run_id, "econ", period, params, now, (label, strategy_version(strategy, rule)))
    rows = []
    for capital in capitals:
        s = replace(strategy, initial_equity=capital)
        bench = {"equal_weight": equal_weight_curve(market.close, market.universe, capital),
                 "liquidity_weighted": liquidity_weighted_curve(market.close, market.universe, market.adv_value,
                                                                capital),
                 "vnindex": capital * market.index_close / market.index_close[0]}
        bench_cagr = {}
        for name, curve in bench.items():
            m = curve_metrics(curve)
            bench_cagr[name] = m["cagr"]
            rows += [(run_id, capital, "gross", name, k, v) for k, v in m.items()]
        for cname, model in COSTS.items():
            res = run(market, s, signal_selector, model)
            m = strategy_metrics(res, capital)
            m["cost_fallbacks"] = float(res.cost_fallbacks)
            cols = {sym: j for j, sym in enumerate(market.symbols)}
            entries = [t for t in res.trades if t.entry_index > 0]
            m |= participation(np.array([t.shares * t.entry_price for t in entries]),
                               np.array([market.adv_value[t.entry_index - 1, cols[t.symbol]] for t in entries]))
            m |= {f"excess_cagr_vs_{b}": m["cagr"] - c for b, c in bench_cagr.items()}
            rows += [(run_id, capital, cname, "strategy", k, v) for k, v in m.items()]
            if cname in CONTROL_COSTS:
                ctl_rows, outcomes = _controls(run_id, market, s, model, keys, res, capital, cname, n_control)
                rows += ctl_rows
                store.con.executemany("INSERT INTO control_outcomes VALUES (?, ?, ?, ?, ?, ?, ?)", outcomes)
    for cname in ("flat", "v1_k1"):                 # valuation policy: write-down of stuck positions
        wd = run(market, replace(strategy, initial_equity=BREAK_EVEN_CAPITAL), signal_selector, COSTS[cname],
                 writedown_after=WRITEDOWN_SESSIONS)
        rows.append((run_id, BREAK_EVEN_CAPITAL, cname, f"writedown_{WRITEDOWN_SESSIONS}", "cagr",
                     curve_metrics(wd.equity)["cagr"]))
    be_strategy = replace(strategy, initial_equity=BREAK_EVEN_CAPITAL)
    for cname, model in (("flat", FLAT_AS_MODEL), ("v1_k1", COSTS["v1_k1"])):
        rows.append((run_id, BREAK_EVEN_CAPITAL, cname, "strategy", "break_even_extra_cost_per_side",
                     break_even(market, be_strategy, model)))
    store.con.executemany("INSERT INTO economic_results VALUES (?, ?, ?, ?, ?, ?)",
                          [(a, b, c, d, e, None if f is None or not np.isfinite(f) else float(f))
                           for a, b, c, d, e, f in rows])
    store.finish_run(run_id)
    return run_id


def _controls(run_id: str, market: Market, s: Strategy, model: CostModel | None, keys: dict, res, capital: float,
              cname: str, n: int) -> tuple[list, list]:
    """Summary rows (share beaten, exceedance p, percentiles) and every control's outcome."""
    out, outcomes = [], []
    final = res.equity[-1]
    for name, make in (("random", lambda seed: random_selector(seed)),
                       ("industry_matched", lambda seed: matched_random_selector(seed, keys["industry"])),
                       ("beta_matched", lambda seed: matched_random_selector(seed, keys["beta"]))):
        runs = [run(market, s, make(seed), model) for seed in range(n)]
        finals = [r.equity[-1] for r in runs]
        cagr = [curve_metrics(r.equity)["cagr"] for r in runs]
        out += [(run_id, capital, cname, name, k, v) for k, v in controls.summary(final, finals, cagr).items()]
        outcomes += controls.outcome_rows(run_id, capital, cname, name, finals, cagr)
    return out, outcomes


def break_even(market: Market, s: Strategy, model: CostModel, iterations: int = 14) -> float:
    """Extra cost per side (on top of `model`) at which the strategy's CAGR equals the equal-weight universe's;
    0 when it does not beat equal weight even without extra cost."""
    target = curve_metrics(equal_weight_curve(market.close, market.universe, s.initial_equity))["cagr"]

    def gap(x: float) -> float:
        return curve_metrics(run(market, s, signal_selector, replace(model, extra_flat=x)).equity)["cagr"] - target
    if gap(0.0) <= 0:
        return 0.0
    lo, hi = 0.0, BREAK_EVEN_MAX
    if gap(hi) > 0:
        return hi
    for _ in range(iterations):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gap(mid) > 0 else (lo, mid)
    return (lo + hi) / 2

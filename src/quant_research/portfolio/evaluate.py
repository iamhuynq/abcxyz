"""`quant portfolio evaluate`: the pre-declared construction grid of docs/portfolio-construction-plan.md.

The matched random control applies the same construction to random scores whose rank persistence between
rebalances equals the signal's (AR(1) per stock), so it trades about as much as the strategy.

Research period only. Each configuration is one run and logs one test (excess daily log return vs the
equal-weight universe at 1 bn VND with cost model v1 k = 1, Newey-West), so the grid counts in the BH family.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime

import numpy as np
import scipy.stats as sps

from quant_research import controls
from quant_research.backtest import runner
from quant_research.backtest.benchmarks import key_matrices, liquidity_weighted_curve
from quant_research.backtest.costs import CostModel
from quant_research.backtest.data import Market, UniverseRule
from quant_research.backtest.metrics import WRITEDOWN_SESSIONS, curve_metrics, equal_weight_curve, participation
from quant_research.econ import FLAT_AS_MODEL
from quant_research.portfolio.construction import Construction
from quant_research.portfolio.engine import PortfolioResult, rebalance_days, run_targets
from quant_research.results import ResearchParams, ResultsStore
from quant_research.stats import mean_test

CAPITALS = (1e9, 10e9, 100e9)
COSTS: dict[str, CostModel | None] = {"flat": None, "v1_k1": CostModel(k=1.0),
                                      "v1_k1_tick": CostModel(k=1.0, spread="tick")}
DECISION = (1e9, "v1_k1")
SELL_LAG = 3                    # T+2 assumption: first sale at the open of b + 3 (see portfolio/engine.py)
SELL_LAG_OPTIMISTIC = 2         # sensitivity, decision setting only
N_CONTROL = controls.N_CONTROL        # 20 before the review of 2026-11-10; recorded runs store theirs
TEST_LAG = 10
BREAK_EVEN_MAX = 0.05


@dataclass(frozen=True)
class Config:
    signal: str                         # Factor Engine factor (rank_pct is the score)
    schedule: str                       # weekly / monthly
    construction: Construction
    cash_when: tuple[str, str] | None = None   # (market_regimes dimension, state) -> hold cash


BASE = Construction(n_names=20, entry_rank=20, exit_rank=20)
BUFFER = replace(BASE, exit_rank=40)
CONFIGS = {
    "C1": Config("order_flow", "weekly", BASE),
    "C2": Config("order_flow", "weekly", BUFFER),
    "C3": Config("order_flow", "monthly", BUFFER),
    "C4": Config("momentum_12_1", "monthly", BASE),
    "C5": Config("momentum_12_1", "monthly", BUFFER),
    "C6": Config("momentum_12_1", "monthly", BUFFER, ("direction", "Bear")),
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolio_results (
    run_id VARCHAR NOT NULL, config VARCHAR NOT NULL, capital DOUBLE NOT NULL, cost_model VARCHAR NOT NULL,
    series VARCHAR NOT NULL, metric VARCHAR NOT NULL, value DOUBLE,
    PRIMARY KEY (run_id, capital, cost_model, series, metric));
"""


class PortfolioRefused(ValueError):
    pass


def config_version(cfg: Config) -> str:
    return hashlib.sha256(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()[:8]


def score_matrix(store: ResultsStore, market: Market, factor: str) -> np.ndarray:
    out = np.full(market.open.shape, np.nan)
    d_index = {d: i for i, d in enumerate(market.dates.tolist())}
    s_index = {s: j for j, s in enumerate(market.symbols.tolist())}
    df = store.con.execute("""SELECT symbol, date, rank_pct FROM rs.stock_factors WHERE factor = ?
                              AND date BETWEEN ? AND ?""",
                           [factor, market.dates[0].item(), market.dates[-1].item()]).df()
    if not df.empty:
        i = np.array([d_index.get(d, -1) for d in df["date"].dt.date], dtype=int)
        j = np.array([s_index.get(s, -1) for s in df["symbol"]], dtype=int)
        ok = (i >= 0) & (j >= 0)
        out[i[ok], j[ok]] = df["rank_pct"].to_numpy(float)[ok]
    return out


def cash_mask(store: ResultsStore, market: Market, cash_when: tuple[str, str] | None) -> np.ndarray | None:
    if cash_when is None:
        return None
    dim, state = cash_when
    rows = dict(store.con.execute(f'SELECT date, "{dim}" FROM rs.market_regimes').fetchall())
    return np.array([rows.get(d) == state for d in market.dates.tolist()])


@dataclass(frozen=True)
class Prepared:
    market: Market
    scores: np.ndarray
    industry: np.ndarray
    cash_on: np.ndarray | None


def prepare(store: ResultsStore, cfg: Config, period: str = "research",
            rule: UniverseRule = UniverseRule()) -> Prepared:
    """The inputs of one configuration: the market from the first scored session, scores, industries, cash rule.
    Shared by `evaluate` and the Risk Engine's re-simulation (one code path; nothing is written)."""
    full = runner._market(store, period, rule)
    scores = score_matrix(store, full, cfg.signal)
    has = np.flatnonzero(np.isfinite(scores).any(axis=1))
    window = slice(int(has[0]) if len(has) else 0, len(full.dates))
    market, scores = runner._slice(full, window), scores[window]
    return Prepared(market, scores, key_matrices(store.con, market)["industry"],
                    cash_mask(store, market, cfg.cash_when))


def simulate(p: Prepared, cfg: Config, capital: float = DECISION[0], costs: CostModel | None = COSTS[DECISION[1]],
             sell_lag: int = SELL_LAG, writedown_after: int | None = None) -> PortfolioResult:
    """One simulation of a configuration (pure: nothing is stored)."""
    return run_targets(p.market, cfg.construction, lambda i: p.scores[i], cfg.schedule, capital, p.industry, p.cash_on,
                       costs, sell_lag, writedown_after)


def already_run(store: ResultsStore, name: str, period: str) -> str | None:
    row = store.con.execute("""SELECT run_id FROM research_runs WHERE status = 'ok' AND kind = 'portfolio'
                               AND pattern_name = ? AND period = ? ORDER BY created_at DESC LIMIT 1""",
                            [name, period]).fetchone()
    return row[0] if row else None


def evaluate(store: ResultsStore, name: str, now: datetime, period: str = "research",
             configs: dict[str, Config] = CONFIGS, rule: UniverseRule = UniverseRule(),
             capitals: tuple[float, ...] = CAPITALS, n_control: int = N_CONTROL) -> str:
    if period != "research":
        raise PortfolioRefused(f"Portfolio evaluation runs on the research period only, not {period!r}.")
    earlier = already_run(store, name, period)
    if earlier:
        raise PortfolioRefused(f"Run {earlier} already logged config {name}; invalidate it with a reason first.")
    cfg = configs[name]
    store.con.execute(SCHEMA)
    store.con.execute(controls.SCHEMA)
    prep = prepare(store, cfg, period, rule)
    market, scores, industry, cash_on = prep.market, prep.scores, prep.industry, prep.cash_on
    run_id = f"{now:%Y%m%dT%H%M%S}-portfolio-{name}-{period}"
    store.start_run(run_id, "portfolio", period, ResearchParams(extra={"config": asdict(cfg), "universe": asdict(rule),
                    "capitals": capitals, "n_control": n_control}), now, (name, config_version(cfg)))

    def simulate(capital: float, costs: CostModel | None, score_fn=lambda i: scores[i],
                 sell_lag: int = SELL_LAG, writedown_after: int | None = None) -> PortfolioResult:
        return run_targets(market, cfg.construction, score_fn, cfg.schedule, capital, industry, cash_on, costs,
                           sell_lag, writedown_after)

    days = rebalance_days(market.dates, cfg.schedule)
    rho = rank_persistence(scores, days)
    rows, decision_curve, ew_curve = [(DECISION[0], "v1_k1", "strategy", "rank_persistence", rho)], None, None
    for capital in capitals:
        bench = {"equal_weight": equal_weight_curve(market.close, market.universe, capital),
                 "liquidity_weighted": liquidity_weighted_curve(market.close, market.universe, market.adv_value,
                                                                capital),
                 "vnindex": capital * market.index_close / market.index_close[0]}
        bench_cagr = {b: curve_metrics(c)["cagr"] for b, c in bench.items()}
        rows += [(capital, "gross", b, "cagr", v) for b, v in bench_cagr.items()]
        for cname, model in COSTS.items():
            res = simulate(capital, model)
            m = _metrics(res, capital)
            m |= participation(np.array([abs(t[2]) * t[3] for t in res.trades]),
                               np.array([market.adv_value[t[0] - 1, t[1]] for t in res.trades]))
            m |= {f"excess_cagr_vs_{b}": m["cagr"] - v for b, v in bench_cagr.items()}
            rows += [(capital, cname, "strategy", k, v) for k, v in m.items()]
            if cname == "v1_k1":
                ctl_rows, outcomes = _control(simulate, capital, model, res, days, rho, market.open.shape[1],
                                              n_control)
                rows += ctl_rows
                store.con.executemany("INSERT INTO control_outcomes VALUES (?, ?, ?, ?, ?, ?, ?)",
                                      [(run_id, *o) for o in outcomes])
            if (capital, cname) == DECISION:
                decision_curve, ew_curve = res.equity, bench["equal_weight"]
    written = simulate(DECISION[0], COSTS[DECISION[1]], writedown_after=WRITEDOWN_SESSIONS)
    rows += [(DECISION[0], DECISION[1], f"writedown_{WRITEDOWN_SESSIONS}", "cagr", curve_metrics(written.equity)["cagr"])]
    optimistic = simulate(DECISION[0], COSTS[DECISION[1]], sell_lag=SELL_LAG_OPTIMISTIC)
    rows += [(DECISION[0], DECISION[1], "sell_lag_2", k, v) for k, v in _metrics(optimistic, DECISION[0]).items()]
    ew_cagr = curve_metrics(equal_weight_curve(market.close, market.universe, DECISION[0]))["cagr"]
    for cname, model in (("flat", FLAT_AS_MODEL), ("v1_k1", COSTS["v1_k1"])):
        rows.append((DECISION[0], cname, "strategy", "break_even_extra_cost_per_side",
                     _break_even(simulate, DECISION[0], model, ew_cagr)))
    rows.append((DECISION[0], DECISION[1], f"writedown_{WRITEDOWN_SESSIONS}", "break_even_extra_cost_per_side",
                 _break_even(simulate, DECISION[0], COSTS[DECISION[1]], ew_cagr, writedown_after=WRITEDOWN_SESSIONS)))
    p = None
    if decision_curve is not None:
        excess = np.diff(np.log(decision_curve)) - np.diff(np.log(ew_curve))
        _, t, p, _, _ = mean_test(excess, TEST_LAG)
        rows += [(DECISION[0], DECISION[1], "test", "excess_vs_equal_weight_daily_log_mean", float(excess.mean())),
                 (DECISION[0], DECISION[1], "test", "excess_vs_equal_weight_t", t),
                 (DECISION[0], DECISION[1], "test", "excess_vs_equal_weight_p", p)]
    store.log_hypothesis(run_id, "portfolio_excess_vs_equal_weight", period, p, now)
    store.con.executemany("INSERT INTO portfolio_results VALUES (?, ?, ?, ?, ?, ?, ?)",
                          [(run_id, name, c, cm, s, k, None if v is None or not np.isfinite(v) else float(v))
                           for c, cm, s, k, v in rows])
    store.finish_run(run_id)
    return run_id


def _metrics(res: PortfolioResult, initial: float) -> dict[str, float]:
    m = curve_metrics(res.equity)
    years = max(len(res.equity) / 250, 1e-9)
    m |= {"turnover_per_year": res.traded_notional / float(res.equity.mean()) / years,
          "costs_paid_x_initial": res.costs_paid / initial,
          "avg_names": float(res.n_names.mean()),
          "avg_cash_share": float(np.mean(res.cash / np.where(res.equity > 0, res.equity, np.nan))),
          "max_target_industry_weight": res.max_target_industry_weight,
          "max_actual_industry_weight": res.max_actual_industry_weight,
          "industry_cap_breach_sessions": float(res.industry_cap_breach_sessions),
          "industry_cap_breach_after_trades": float(res.industry_cap_breach_after_trades),
          "orders_capped": float(res.orders_capped), "orders_blocked": float(res.orders_blocked),
          "orders_blocked_no_adv": float(res.orders_blocked_no_adv), "orders_blocked_t2": float(res.orders_blocked_t2),
          "orders_partial_t2": float(res.orders_partial_t2), "stale_value_share_mean": res.stale_value_share_mean,
          "stale_value_share_max": res.stale_value_share_max,
          "cost_fallbacks": float(res.cost_fallbacks),
          "data_error_exits": float(res.data_error_exits)}
    return m


def rank_persistence(scores: np.ndarray, days: np.ndarray, min_stocks: int = 10) -> float:
    """Mean Spearman correlation of the scores between consecutive rebalance dates (stocks scored on both)."""
    out = []
    for a, b in zip(days[:-1], days[1:]):
        ok = np.isfinite(scores[a]) & np.isfinite(scores[b])
        if ok.sum() >= min_stocks:
            out.append(sps.spearmanr(scores[a, ok], scores[b, ok]).statistic)
    return float(np.nanmean(out)) if out else 0.0


def persistent_random_scores(seed: int, k: int, n_days: int, rho_spearman: float) -> np.ndarray:
    """(n_days x k) AR(1) normal scores per column whose rank correlation between consecutive rebalances is
    rho_spearman (Pearson rho = 2 sin(pi rho_s / 6) for a bivariate normal)."""
    rho = float(np.clip(2 * np.sin(np.pi * rho_spearman / 6), -0.999, 0.999))
    rng = np.random.default_rng(seed)
    z = np.empty((n_days, k))
    z[0] = rng.standard_normal(k)
    for r in range(1, n_days):
        z[r] = rho * z[r - 1] + np.sqrt(1 - rho * rho) * rng.standard_normal(k)
    return z


def _control(simulate, capital: float, model: CostModel | None, res: PortfolioResult, days: np.ndarray, rho: float,
             k: int, n: int) -> tuple[list, list]:
    """Same construction on random scores with the signal's own rank persistence between rebalances. Returns
    the summary rows and every control's outcome (capital, cost model, series, seed, final equity, CAGR)."""
    pos = {int(d): r for r, d in enumerate(days)}
    runs = []
    for seed in range(n):
        z = persistent_random_scores(seed, k, len(days), rho)
        runs.append(simulate(capital, model, lambda i, z=z: z[pos[i]]))
    finals = [r.equity[-1] for r in runs]
    cagr = [curve_metrics(r.equity)["cagr"] for r in runs]
    turnover = np.array([_metrics(r, capital)["turnover_per_year"] for r in runs], dtype=float)
    series = "random_construction"
    rows = [(capital, "v1_k1", series, k, v) for k, v in controls.summary(res.equity[-1], finals, cagr).items()]
    rows.append((capital, "v1_k1", series, "median_turnover_per_year", float(np.nanmedian(turnover))))
    outcomes = [o[1:] for o in controls.outcome_rows("", capital, "v1_k1", series, finals, cagr)]
    return rows, outcomes


def _break_even(simulate, capital: float, model: CostModel, target: float, iterations: int = 14,
                writedown_after: int | None = None) -> float:
    def gap(x: float) -> float:
        res = simulate(capital, replace(model, extra_flat=x), writedown_after=writedown_after)
        return curve_metrics(res.equity)["cagr"] - target
    if gap(0.0) <= 0:
        return 0.0
    if gap(BREAK_EVEN_MAX) > 0:
        return BREAK_EVEN_MAX
    lo, hi = 0.0, BREAK_EVEN_MAX
    for _ in range(iterations):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gap(mid) > 0 else (lo, mid)
    return (lo + hi) / 2

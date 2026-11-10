"""Report of an economic evaluation (`quant econ evaluate`)."""

import pandas as pd

from quant_research.backtest.metrics import WRITEDOWN_SESSIONS
from quant_research.econ import BREAK_EVEN_CAPITAL, COSTS, CONTROL_COSTS
from quant_research.provenance import from_runs
from quant_research.results import ResultsStore


def _pct(v) -> str:
    return "-" if v is None or pd.isna(v) else f"{v:+.1%}"


def _num(v, fmt: str = ".2f") -> str:
    return "-" if v is None or pd.isna(v) else format(v, fmt)


def render(store: ResultsStore, run_id: str) -> str:
    df = store.con.execute("SELECT * FROM economic_results WHERE run_id = ?", [run_id]).df()
    get = df.set_index(["capital", "cost_model", "series", "metric"])["value"]

    def v(capital, cost, series, metric):
        return get.get((capital, cost, series, metric))

    capitals = sorted(df["capital"].unique())
    lines = [f"# Economic evaluation {run_id}", "", *from_runs(store.con, [run_id]), "",
             "Research period only; descriptive (not in the hypothesis log). Cost models: `flat` = commission "
             "0.15% per side + sell tax 0.10% (the backtest default); `v1_k*` = flat + half-spread (max of tick "
             "floor and the Abdi-Ranaldo CHL estimate) + impact k x sigma_20 x sqrt(order / ADV20), per side; "
             "`v1_k1_tick` uses the tick floor only (lower bound: the 21-session CHL estimate is noisy and the max "
             "overstates spreads of liquid stocks). "
             "Benchmarks are gross (no costs).", "",
             "## Strategy by capital and cost model", "",
             "| Capital (bn VND) | Costs | CAGR | Sharpe | Max DD | Turnover / yr | Costs paid (x initial) | "
             "vs equal weight | vs liquidity-weighted | vs VNINDEX |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in capitals:
        for cost in COSTS:
            lines.append(f"| {c / 1e9:g} | {cost} | {_pct(v(c, cost, 'strategy', 'cagr'))} | "
                         f"{_num(v(c, cost, 'strategy', 'sharpe'))} | {_pct(v(c, cost, 'strategy', 'max_drawdown'))} | "
                         f"{_num(v(c, cost, 'strategy', 'turnover_per_year'), '.1f')} | "
                         f"{_num(v(c, cost, 'strategy', 'costs_paid_pct_of_initial'), '.2f')} | "
                         f"{_pct(v(c, cost, 'strategy', 'excess_cagr_vs_equal_weight'))} | "
                         f"{_pct(v(c, cost, 'strategy', 'excess_cagr_vs_liquidity_weighted'))} | "
                         f"{_pct(v(c, cost, 'strategy', 'excess_cagr_vs_vnindex'))} |")
    lines += ["", "Order size: order value / ADV20 at the signal close, median and 95th percentile (v1 k = 1):", ""]
    lines += [f"- {c / 1e9:g} bn VND: median {_num(v(c, 'v1_k1', 'strategy', 'participation_median'), '.2%')}, "
              f"p95 {_num(v(c, 'v1_k1', 'strategy', 'participation_p95'), '.2%')}" for c in capitals]
    lines += ["", f"Valuation policy (2026-10-09): stuck positions written down to 0 after {WRITEDOWN_SESSIONS} sessions "
              f"without a trade, {BREAK_EVEN_CAPITAL / 1e9:g} bn VND: CAGR flat "
              f"{_pct(v(BREAK_EVEN_CAPITAL, 'flat', f'writedown_{WRITEDOWN_SESSIONS}', 'cagr'))}, v1 "
              f"{_pct(v(BREAK_EVEN_CAPITAL, 'v1_k1', f'writedown_{WRITEDOWN_SESSIONS}', 'cagr'))}. A pre-registered rule "
              "is judged on the worse of the two valuations."]
    lines += ["", "## Benchmarks (gross)", "", "| Capital (bn VND) | Equal weight | Liquidity-weighted | VNINDEX |",
              "|---|---|---|---|"]
    for c in capitals:
        lines.append(f"| {c / 1e9:g} | {_pct(v(c, 'gross', 'equal_weight', 'cagr'))} | "
                     f"{_pct(v(c, 'gross', 'liquidity_weighted', 'cagr'))} | {_pct(v(c, 'gross', 'vnindex', 'cagr'))} |")
    lines += ["", "## Controls (20 seeds each, same mechanics and costs)", "",
              "Random = a random tenth of the universe each day; matched = each candidate replaced by a random "
              "universe stock of the same industry (ICB level 2) or beta quintile.", "",
              "Note: the liquidity-weighted benchmark (weights = ADV20, rebalanced daily) is a poor capitalization "
              "proxy: it overweights stocks right after volume spikes. Compare with VNINDEX for a cap-weighted view.",
              "",
              "Exceedance p = (1 + controls ending at or above the strategy) / (1 + controls); runs before the "
              "review of 2026-11-10 used 20 controls and have no exceedance p.", "",
              "| Capital (bn VND) | Costs | Control | Share of runs beaten | Exceedance p | Control median CAGR [p5, p95] |",
              "|---|---|---|---|---|---|"]
    for c in capitals:
        for cost in CONTROL_COSTS:
            for ctl in ("random", "industry_matched", "beta_matched"):
                lines.append(f"| {c / 1e9:g} | {cost} | {ctl} | {_num(v(c, cost, ctl, 'share_beaten'), '.0%')} | "
                             f"{_num(v(c, cost, ctl, 'exceedance_p'), '.3f')} | "
                             f"{_pct(v(c, cost, ctl, 'median_cagr'))} [{_pct(v(c, cost, ctl, 'p05_cagr'))}, "
                             f"{_pct(v(c, cost, ctl, 'p95_cagr'))}] |")
    be_flat = v(BREAK_EVEN_CAPITAL, "flat", "strategy", "break_even_extra_cost_per_side")
    be_v1 = v(BREAK_EVEN_CAPITAL, "v1_k1", "strategy", "break_even_extra_cost_per_side")
    fallbacks = v(BREAK_EVEN_CAPITAL, "v1_k1", "strategy", "cost_fallbacks")
    lines += ["", "## Break-even", "",
              f"Extra cost per side, at {BREAK_EVEN_CAPITAL / 1e9:g} bn VND, at which the strategy's CAGR falls to the "
              "equal-weight universe's (0 = it does not beat equal weight even without extra cost):", "",
              f"- on top of `flat` costs: **{_num(be_flat, '.3%')}** (the spread + impact the strategy can absorb);",
              f"- on top of `v1_k1`: **{_num(be_v1, '.3%')}**.", "",
              f"Trades priced with a fallback input (no spread / volatility / ADV estimate): {_num(fallbacks, '.0f')} "
              "at 1 bn VND.", "",
              "Marking: positions at their last traded close (a no-trade session keeps the previous mark). Invested "
              "value marked with a price older than 5 sessions at 1 bn VND, v1: mean "
              f"{_num(v(BREAK_EVEN_CAPITAL, 'v1_k1', 'strategy', 'stale_value_share_mean'), '.1%')}, max "
              f"{_num(v(BREAK_EVEN_CAPITAL, 'v1_k1', 'strategy', 'stale_value_share_max'), '.1%')}; entries blocked "
              f"for a missing ADV: {_num(v(BREAK_EVEN_CAPITAL, 'v1_k1', 'strategy', 'entries_blocked_no_adv'), '.0f')}."]
    return "\n".join(lines)

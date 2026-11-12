"""Stress scenarios (docs/risk-engine-pr2-plan.md, decisions D2 to D8). Pure functions on an assessed snapshot.

Scenarios are illustrations, not forecasts or probabilities, and not policy limits: a result is MEASURED or
INCOMPLETE, never PASS / BREACH. Scenarios are never summed. A valuation loss (positions marked down) and a
liquidation cost (selling) are separate numbers. A loss is a positive amount of VND lost; a negative loss is a
gain (for example a negative beta). Idiosyncratic return is 0 in every scenario.

Inputs are the positions an assessment returns (value, weight, beta, industry, cluster, ADV, half-spread, sigma,
stale sessions): the same snapshot and conventions as the limits. If the positions cannot be trusted
(unvalued positions, NAV not reconciled), every scenario is INCOMPLETE.
"""

import json
import math
from dataclasses import asdict, dataclass

import pandas as pd

from quant_research.backtest.costs import CostModel
from quant_research.backtest.metrics import WRITEDOWN_SESSIONS
from quant_research.risk import liquidity

MEASURED, INCOMPLETE = "MEASURED", "INCOMPLETE"


@dataclass(frozen=True)
class ScenarioSet:
    set_id: str
    version: str
    market_shocks: tuple[float, ...]
    group_shocks: tuple[float, ...]
    adv_factor: float
    suspension_top: int
    suspension_sessions: tuple[int, ...]
    note: str = ""

    @property
    def key(self) -> str:
        return f"{self.set_id}/{self.version}"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


STRESS_V1 = ScenarioSet(
    "stress", "v1", market_shocks=(-0.05, -0.10, -0.15), group_shocks=(-0.10, -0.20), adv_factor=0.5,
    suspension_top=3, suspension_sessions=(20, WRITEDOWN_SESSIONS + 1),
    note="Declared 2026-11-12 before any stress run. Market shocks: research-period VNINDEX tails (1-day 1% "
         "quantile -4.4%, 5-day -11.0%). Group shocks -10% / -20%. ADV halved. Suspension: the 3 largest positions "
         "for 20 sessions (exit blocked, still at the mark) and 61 (past the 60-session write-down).")
SCENARIO_SETS = {STRESS_V1.key: STRESS_V1}


@dataclass(frozen=True)
class StressResult:
    scenario_id: str
    family: str
    group: str                      # "" for portfolio-wide scenarios
    status: str
    nav_before: float
    valuation_loss: float | None    # VND lost by marking positions down
    cost: float | None              # liquidation cost under the scenario (liquidity family)
    cost_before: float | None       # the same cost without the scenario
    coverage: float | None          # NAV weight of the positions the scenario uses
    affected: tuple[str, ...] = ()
    reason: str = ""
    notes: str = ""

    @property
    def valuation_loss_pct(self) -> float | None:
        return None if self.valuation_loss is None else self.valuation_loss / self.nav_before


def _pct(x: float) -> str:
    return f"{x:+.0%}"


def _missing(pos: pd.DataFrame, col: str) -> tuple[str, ...]:
    return tuple(sorted(pos.loc[pos[col].isna(), "symbol"]))


def market(pos: pd.DataFrame, nav: float, shocks: tuple[float, ...]) -> list[StressResult]:
    """Beta-adjusted (sum of value x beta x shock; INCOMPLETE if any beta is missing) and uniform (every position
    moves by the shock; not beta-adjusted)."""
    out, lacking = [], _missing(pos, "beta")
    invested = float(pos["value"].sum())
    for s in shocks:
        if lacking:
            out.append(StressResult(f"market_beta_{_pct(s)}", "market", "", INCOMPLETE, nav, None, None, None,
                                    float(pos.loc[pos["beta"].notna(), "weight"].sum()), lacking,
                                    f"no beta for {', '.join(lacking)}"))
        else:
            loss = -float((pos["value"] * pos["beta"]).sum()) * s
            out.append(StressResult(f"market_beta_{_pct(s)}", "market", "", MEASURED, nav, loss, None, None,
                                    invested / nav, notes=f"each position moves by its beta x {s:+.0%}"))
        out.append(StressResult(f"market_uniform_{_pct(s)}", "market", "", MEASURED, nav, -invested * s, None, None,
                                invested / nav, notes=f"every position moves by {s:+.0%}; not beta-adjusted"))
    return out


def groups(pos: pd.DataFrame, nav: float, shocks: tuple[float, ...], col: str, family: str) -> list[StressResult]:
    """One group at a time, on that group's positions only. Holdings without a label could belong to any group,
    so every result of the family is INCOMPLETE while one exists (the shown loss covers labelled holdings)."""
    lacking = _missing(pos, col)
    why = (f"{pos.loc[pos[col].isna(), 'weight'].sum():.1%} of NAV has no {col}: {', '.join(lacking)}; the loss "
           f"covers labelled holdings only") if lacking else ""
    out = []
    labelled = pos[pos[col].notna()]
    for s in shocks:
        for g, part in sorted(labelled.groupby(col), key=lambda kv: (-kv[1]["value"].sum(), str(kv[0]))):
            out.append(StressResult(f"{family}_{_pct(s)}", family, str(g), INCOMPLETE if lacking else MEASURED, nav,
                                    -float(part["value"].sum()) * s, None, None, float(part["weight"].sum()),
                                    tuple(sorted(part["symbol"])), why))
        if not len(labelled):
            out.append(StressResult(f"{family}_{_pct(s)}", family, "", INCOMPLETE, nav, None, None, None, 0.0,
                                    lacking, why or f"no position has a {col}"))
    return out


def liquidity_shock(pos: pd.DataFrame, nav: float, factor: float, params: liquidity.LiquidityParams,
                    model: CostModel = CostModel()) -> StressResult:
    """ADV x factor, through the same liquidity model as the limits (cost model v1, fee and tax)."""
    sid = f"liquidity_adv_x{factor:g}"
    lacking = _missing(pos, "adv")
    before = float(pos["liquidation_cost"].sum()) if not lacking else None
    if lacking:
        return StressResult(sid, "liquidity", "", INCOMPLETE, nav, 0.0, None, None,
                            float(pos.loc[pos["adv"].notna(), "weight"].sum()), lacking,
                            f"ADV20 missing or invalid for {', '.join(lacking)}")
    rows = [liquidity.position(r.symbol, r.value, r.adv * factor, None if pd.isna(r.half_spread) else r.half_spread,
                               None if pd.isna(r.sigma) else r.sigma, 0, False, params, model)
            for r in pos.itertuples()]
    cost = sum(x.cost for x in rows)
    days = max((x.days for x in rows), default=0.0)
    ill = sum(x.value for x in rows if x.illiquid) / nav
    ill_before = float(pos.loc[pos["illiquid"].fillna(False).astype(bool), "weight"].sum())
    fb = sum(x.value for x in rows if x.cost_fallback) / nav
    return StressResult(sid, "liquidity", "", MEASURED, nav, 0.0, cost, before, float(pos["weight"].sum()),
                        tuple(sorted(x.symbol for x in rows if x.illiquid)),
                        notes=f"max liquidation days {pos['liquidation_days'].max():.3g} -> {days:.3g}; illiquid weight "
                              f"{ill_before:.1%} -> {ill:.1%}; cost on fallback inputs {fb:.1%} of NAV")


def suspension(pos: pd.DataFrame, nav: float, top: int, sessions: tuple[int, ...]) -> list[StressResult]:
    """The `top` largest positions by NAV weight (ties by symbol) cannot trade. Up to the 60-session write-down
    they stay at their mark (no valuation loss; the exit is blocked); past it they are valued at 0."""
    chosen = pos.sort_values(["weight", "symbol"], ascending=[False, True]).head(top)
    blocked = float(chosen["weight"].sum())
    out = []
    for n in sessions:
        written = n > WRITEDOWN_SESSIONS
        loss = float(chosen["value"].sum()) if written else 0.0
        note = (f"no trade for {n} sessions: valued at 0 (write-down after {WRITEDOWN_SESSIONS})" if written else
                f"exit blocked for {n} sessions ({blocked:.1%} of NAV cannot be sold); still valued at the mark, "
                f"since the write-down starts after {WRITEDOWN_SESSIONS}")
        out.append(StressResult(f"suspension_top{top}_{n}s", "suspension", "", MEASURED, nav, loss, None, None,
                                blocked, tuple(chosen["symbol"]), notes=note))
    return out


def run(pos: pd.DataFrame, nav: float, issues: list[str], scen: ScenarioSet = STRESS_V1,
        params: liquidity.LiquidityParams = liquidity.LiquidityParams()) -> list[StressResult]:
    if not (nav and math.isfinite(nav) and nav > 0):
        return [StressResult("all", "all", "", INCOMPLETE, nav, None, None, None, None, (),
                             "NAV missing or not positive")]
    results = (market(pos, nav, scen.market_shocks)
               + groups(pos, nav, scen.group_shocks, "industry", "industry")
               + groups(pos, nav, scen.group_shocks, "cluster", "cluster")
               + [liquidity_shock(pos, nav, scen.adv_factor, params)]
               + suspension(pos, nav, scen.suspension_top, scen.suspension_sessions))
    if issues:          # the positions cannot be trusted: nothing computed on them is a measurement
        why = "; ".join(issues)
        results = [StressResult(r.scenario_id, r.family, r.group, INCOMPLETE, r.nav_before, r.valuation_loss, r.cost,
                                r.cost_before, r.coverage, r.affected, why + (f"; {r.reason}" if r.reason else ""),
                                r.notes) for r in results]
    return results

"""Liquidity risk per position (docs/risk-engine-plan.md, phase D). Pure functions.

A position is sold at most `participation` x ADV20 per session. Its liquidation horizon is value / (that
amount); its cost is the strategy's fee and sell tax plus cost model v1 (half-spread and square-root impact)
on each session's slice. Invalid inputs never mean "liquid": an ADV that is missing, zero, negative or not
finite gives no horizon and no cost (unknown), and the position counts as missing data. A position without a
trade for more than `stale_sessions`, or closing at the floor on the valuation date, cannot be assumed
sellable: its estimate is marked conditional.
"""

import math
from dataclasses import dataclass

from quant_research.backtest.costs import CostModel
from quant_research.backtest.engine import Strategy


@dataclass(frozen=True)
class LiquidityParams:
    participation: float = 0.10        # share of ADV20 sold per session
    illiquid_days: float = 1.0         # a position needing more sessions than this is illiquid
    stale_sessions: int = 5            # sessions without a trade before a mark is stale (engines' STALE_SESSIONS)


@dataclass(frozen=True)
class PositionLiquidity:
    symbol: str
    value: float
    adv: float | None                  # valid ADV20 (VND) or None
    days: float | None                 # sessions to liquidate at the participation rate; None if ADV invalid
    illiquid: bool | None              # None if unknown
    cost: float | None                 # estimated liquidation cost (VND); None if unknown
    cost_fallback: bool                # the cost model used a fallback input (missing spread or volatility)
    stale_sessions: int | None         # sessions since the last trade at the valuation date
    stale: bool | None
    conditional: bool                  # stale, or at the floor: the estimate assumes trading resumes
    reason: str                        # "" or why the estimate is unknown / conditional


def valid_adv(adv) -> float | None:
    try:
        a = float(adv)
    except (TypeError, ValueError):
        return None
    return a if math.isfinite(a) and a > 0 else None


def liquidation_cost(value: float, adv: float, half_spread, sigma, params: LiquidityParams,
                     model: CostModel = CostModel(), strategy: Strategy = Strategy()) -> tuple[float, bool]:
    """(cost in VND, any fallback) of selling `value` in slices of participation x ADV per session."""
    per_day = params.participation * adv
    half_spread = float("nan") if half_spread is None else float(half_spread)   # missing input: the model's fallback
    sigma = float("nan") if sigma is None else float(sigma)
    remaining, cost, fallback = value, 0.0, False
    while remaining > 1e-9:
        piece = min(per_day, remaining)
        extra, fb = model.side(half_spread, sigma, adv, piece)
        cost += piece * (strategy.fee + strategy.sell_tax + extra)
        fallback |= fb
        remaining -= piece
    return cost, fallback


def position(symbol: str, value: float, adv, half_spread, sigma, stale_sessions: int | None, at_floor: bool,
             params: LiquidityParams = LiquidityParams(), model: CostModel = CostModel()) -> PositionLiquidity:
    a = valid_adv(adv)
    stale = None if stale_sessions is None else stale_sessions > params.stale_sessions
    reasons = []
    if a is None:
        reasons.append("ADV20 missing or invalid")
    if stale_sessions is None:
        reasons.append("no trade found")
    elif stale:
        reasons.append(f"no trade for {stale_sessions} sessions")
    if at_floor:
        reasons.append("closed at the floor")
    if a is None:
        return PositionLiquidity(symbol, value, None, None, None, None, False, stale_sessions, stale,
                                 bool(stale) or at_floor, "; ".join(reasons))
    days = value / (params.participation * a)
    cost, fb = liquidation_cost(value, a, half_spread, sigma, params, model)
    return PositionLiquidity(symbol, value, a, days, days > params.illiquid_days, cost, fb, stale_sessions, stale,
                             bool(stale) or at_floor or stale_sessions is None, "; ".join(reasons))

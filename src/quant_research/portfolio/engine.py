"""Target-weight engine (docs/portfolio-construction-plan.md).

At each rebalance close t the targets are computed from data known at t; trades happen at the open of t + 1:
sells first, then buys with the cash available. Rules:
- an order is capped at max_participation x ADV20(t); the rest of the move is skipped. Without a valid ADV
  (NaN, infinite or <= 0) the order is blocked, never uncapped (counted in orders_blocked_no_adv);
- no buy at an open at the ceiling, no trade in a stock without a trade or an open price;
- no sell when the session closes at the floor (the open has no floor flag; approximation);
- T+2 per lot (backtest assumption): shares bought at the open of b arrive in the afternoon of b + 2, and this
  engine trades at opens only, so a lot can be sold from the open of b + sell_lag (sell_lag = 3 by default;
  2 is the optimistic sensitivity). A sale uses eligible lots oldest first; a top-up never locks older
  shares. Orders with no eligible share: orders_blocked_t2; orders reduced to the eligible shares:
  orders_partial_t2;
- a held stock with a data-error jump is sold at its last valid close, with the same costs as any sale.
Costs: commission on both sides, sell tax, plus cost model v1 when given (inputs of the decision close; for a
data-error exit at the close of i, inputs of the close of i - 1).
Between rebalances, positions drift with prices. Positions are marked at the last TRADED close: a session
without trades keeps the previous mark (its close is only the reference price). stale_value_share_* report the
share of portfolio value marked with a price older than STALE_SESSIONS. Industry weights are reported twice:
of the targets, and actual (market value at every close), with the sessions above the cap.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from quant_research.backtest.costs import CostModel
from quant_research.backtest.data import Market
from quant_research.portfolio.construction import Construction, ranks, select, target_weights

FEE, SELL_TAX = 0.0015, 0.001
STALE_SESSIONS = 5
Scores = Callable[[int], np.ndarray]     # rebalance close index -> score per column (NaN = no score)


@dataclass
class PortfolioResult:
    equity: np.ndarray
    cash: np.ndarray
    n_names: np.ndarray
    traded_notional: float = 0.0
    costs_paid: float = 0.0
    cost_fallbacks: int = 0
    orders_capped: int = 0
    orders_blocked: int = 0              # all blocked orders, including the two kinds below
    orders_blocked_no_adv: int = 0
    orders_blocked_t2: int = 0
    orders_partial_t2: int = 0
    stale_value_share_max: float = 0.0
    stale_value_share_mean: float = 0.0     # over sessions with holdings
    data_error_exits: int = 0
    max_target_industry_weight: float = 0.0
    max_actual_industry_weight: float = 0.0
    industry_cap_breach_sessions: int = 0           # closes with an actual industry weight above the cap
    industry_cap_breach_after_trades: int = 0       # of which: the close of a trading session
    trades: list[tuple] = field(default_factory=list)     # (session, column, shares (+ buy / - sell), price, cost)
    # The engine's own state at the close of session `snapshot_at` (run_targets), when asked: shares, marks (the
    # valuation the equity uses), last traded session, cash and equity. The Risk Engine reads it instead of
    # re-deriving the marks.
    state: dict | None = None


def rebalance_days(dates: np.ndarray, schedule: str) -> np.ndarray:
    """Indices of the first session of each week or month (the decision closes)."""
    if schedule == "weekly":
        period = (dates.astype("datetime64[D]").astype(int) + 3) // 7           # Monday-based weeks
    elif schedule == "monthly":
        period = dates.astype("datetime64[M]").astype(int)
    else:
        raise ValueError(f"unknown schedule {schedule!r}")
    return np.flatnonzero(np.r_[True, period[1:] != period[:-1]])


def run_targets(market: Market, c: Construction, scores: Scores, schedule: str, initial: float,
                industry: np.ndarray, cash_on: np.ndarray | None = None, costs: CostModel | None = None,
                sell_lag: int = 3, writedown_after: int | None = None,
                snapshot_at: int | None = None) -> PortfolioResult:
    """industry: (dates x symbols) object keys; cash_on: bool per date, True = hold cash after that close.
    writedown_after: value a position at 0 while it has not traded for more than this many sessions (valuation
    sensitivity for stocks that stop trading; the position can still be sold if it trades again)."""
    n, k = market.open.shape
    res = PortfolioResult(np.zeros(n), np.zeros(n), np.zeros(n, dtype=int))
    shares = np.zeros(k)
    lots: dict[int, list[list[float]]] = {}                       # column -> [[buy session, shares], ...]
    last_price = np.full(k, np.nan)                               # last TRADED close (the mark)
    last_trade = np.full(k, -1)
    stale_sum, held_sessions = 0.0, 0
    cash = initial
    is_rebalance = np.zeros(n, bool)
    is_rebalance[rebalance_days(market.dates, schedule)] = True
    targets: dict[int, float] | None = None
    decision = -1
    for i in range(n):
        traded_today = targets is not None
        if targets is not None:                                   # open of the session after a rebalance close
            marks = last_price if writedown_after is None else np.where(i - last_trade > writedown_after, 0.0,
                                                                         last_price)
            cash = _trade(market, c, res, i, decision, targets, shares, lots, marks, cash, costs, sell_lag)
            targets = None
        for j in np.flatnonzero(shares > 0):                      # close: data errors, then mark to market
            if market.price_jump[i, j]:
                value = shares[j] * last_price[j]
                cost = value * (FEE + SELL_TAX + _extra(res, market, costs, max(i - 1, 0), j, value))
                cash += value - cost
                res.costs_paid += cost
                res.traded_notional += value
                res.trades.append((i, int(j), -shares[j], last_price[j], cost))
                res.data_error_exits += 1
                shares[j] = 0.0
                lots.pop(int(j), None)
            elif market.traded[i, j] and np.isfinite(market.close[i, j]):
                last_price[j] = market.close[i, j]
                last_trade[j] = i
        marks = last_price if writedown_after is None else np.where(i - last_trade > writedown_after, 0.0, last_price)
        held_value = float(np.nansum(shares * marks))
        if held_value > 0:
            # same marks in numerator and denominator (a written-down position counts 0 in both)
            stale = float(np.nansum(np.where(i - last_trade > STALE_SESSIONS, shares * marks, 0.0))) / held_value
            res.stale_value_share_max = max(res.stale_value_share_max, stale)
            stale_sum, held_sessions = stale_sum + stale, held_sessions + 1
        res.equity[i], res.cash[i], res.n_names[i] = cash + held_value, cash, int((shares > 0).sum())
        if i == snapshot_at:
            res.state = {"session": i, "shares": shares.copy(), "marks": np.array(marks, dtype=float),
                         "last_trade": last_trade.copy(), "cash": float(cash), "equity": float(res.equity[i])}
        actual = _max_industry(shares * last_price, industry[i], res.equity[i])
        res.max_actual_industry_weight = max(res.max_actual_industry_weight, actual)
        if actual > c.max_industry_weight + 1e-9:
            res.industry_cap_breach_sessions += 1
            res.industry_cap_breach_after_trades += traded_today
        if is_rebalance[i] and i < n - 1:
            decision = i
            if cash_on is not None and cash_on[i]:
                targets = {}
            else:
                eligible = market.universe[i] & (np.nan_to_num(market.adv_value[i]) > c.min_adv_value)
                chosen = select(ranks(scores(i), eligible), set(np.flatnonzero(shares > 0).tolist()), c)
                sigma = market.sigma[i] if market.sigma is not None else np.full(k, np.nan)
                targets = target_weights(chosen, c, sigma, industry[i])
                weights = np.zeros(k)
                weights[list(targets)] = list(targets.values())
                res.max_target_industry_weight = max(res.max_target_industry_weight,
                                                     _max_industry(weights, industry[i], 1.0))
    res.stale_value_share_mean = stale_sum / held_sessions if held_sessions else 0.0
    return res


def _max_industry(values: np.ndarray, industry_row: np.ndarray, total: float) -> float:
    """Largest industry share of `total` (stocks without an industry are not grouped)."""
    if total <= 0:
        return 0.0
    sums: dict = {}
    for j in np.flatnonzero(np.nan_to_num(values) > 0):
        key = industry_row[j]
        if key is not None:
            sums[key] = sums.get(key, 0.0) + float(values[j])
    return max(sums.values(), default=0.0) / total


def _extra(res: PortfolioResult, market: Market, costs: CostModel | None, t: int, j: int, value: float) -> float:
    if costs is None:
        return 0.0
    if costs.spread == "none":
        hs = 0.0
    else:
        arr = market.tick_half_spread if costs.spread == "tick" else market.half_spread
        hs = arr[t, j] if arr is not None else np.nan
    sigma = market.sigma[t, j] if market.sigma is not None else np.nan
    v, fallback = costs.side(hs, sigma, market.adv_value[t, j], value)
    res.cost_fallbacks += fallback
    return v


def _trade(market: Market, c: Construction, res: PortfolioResult, i: int, t: int, targets: dict[int, float],
           shares: np.ndarray, lots: dict[int, list[list[float]]], last_price: np.ndarray, cash: float,
           costs: CostModel | None, sell_lag: int) -> float:
    price = market.open[i]
    equity = cash + float(np.nansum(shares * last_price))           # value at the decision close
    wanted = {j: targets.get(j, 0.0) * equity for j in set(targets) | set(np.flatnonzero(shares > 0).tolist())}
    adv = market.adv_value[t]
    adv_ok = np.isfinite(adv) & (adv > 0)
    cap = c.max_participation * np.where(adv_ok, adv, 0.0)          # no valid ADV: blocked, never uncapped
    tradable = market.traded[i] & np.isfinite(price) & (price > 0)
    buys = []
    for j, target in sorted(wanted.items()):
        current = shares[j] * price[j] if tradable[j] else np.nan
        if not np.isfinite(current):
            if target != shares[j] * last_price[j]:
                res.orders_blocked += 1
            continue
        delta = target - current
        if delta != 0 and not adv_ok[j]:
            res.orders_blocked += 1
            res.orders_blocked_no_adv += 1
            continue
        if delta < 0:                                               # sells first
            if market.limit_down[i, j]:
                res.orders_blocked += 1
                continue
            eligible = sum(sh for b, sh in lots.get(j, []) if i - b >= sell_lag)
            if eligible <= 0:
                res.orders_blocked += 1
                res.orders_blocked_t2 += 1
                continue
            value = min(-delta, cap[j], current)
            res.orders_capped += value < -delta - 1e-9
            if value > eligible * price[j]:
                value = eligible * price[j]
                res.orders_partial_t2 += 1
            cost = value * (FEE + SELL_TAX + _extra(res, market, costs, t, j, value))
            cash += value - cost
            _consume_lots(lots, j, value / price[j])
            shares[j] -= value / price[j]
            if shares[j] * price[j] < 1e-6:
                shares[j] = 0.0
                lots.pop(j, None)
            res.costs_paid += cost
            res.traded_notional += value
            res.trades.append((i, int(j), -value / price[j], price[j], cost))
        elif delta > 0:
            if market.open_limit_up[i, j]:
                res.orders_blocked += 1
                continue
            buys.append((j, min(delta, cap[j])))
            res.orders_capped += cap[j] < delta - 1e-9
    for j, value in buys:
        rate = FEE + _extra(res, market, costs, t, j, value)
        value = min(value, cash / (1 + rate))
        if value <= 0:
            continue
        cost = value * rate
        cash -= value + cost
        shares[j] += value / price[j]
        lots.setdefault(j, []).append([i, value / price[j]])
        last_price[j] = price[j] if not np.isfinite(last_price[j]) else last_price[j]
        res.costs_paid += cost
        res.traded_notional += value
        res.trades.append((i, int(j), value / price[j], price[j], cost))
    return cash


def _consume_lots(lots: dict[int, list[list[float]]], j: int, qty: float) -> None:
    """Remove `qty` shares from the oldest lots first (only eligible lots are old enough to reach here)."""
    remaining = qty
    for lot in lots.get(j, []):
        take = min(lot[1], remaining)
        lot[1] -= take
        remaining -= take
        if remaining <= 1e-12:
            break
    lots[j] = [lot for lot in lots.get(j, []) if lot[1] > 1e-12]

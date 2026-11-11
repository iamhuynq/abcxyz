"""Portfolio risk metrics (docs/risk-engine-plan.md, phase B). Pure functions on arrays; no database access.

Conventions: daily simple returns of a net equity curve (costs included); annualization by sqrt(250), as in
backtest/metrics.py; a loss is a positive number in VaR / ES. Every function returns None (not 0, not NaN)
when its input is too short, so a missing value can never pass a limit.
"""

import math
from dataclasses import dataclass

import numpy as np

from quant_research.backtest.metrics import SESSIONS_PER_YEAR

MIN_OBS_VAR = {0.95: 250, 0.99: 500}       # returns needed before VaR / ES at that level is reported


@dataclass(frozen=True)
class Drawdown:
    current: float                  # equity[-1] / running peak - 1 (<= 0)
    maximum: float                  # most negative drawdown over the window (<= 0)
    peak_index: int                 # session index of the peak before the maximum drawdown
    trough_index: int               # session index of the maximum drawdown
    current_peak_index: int         # session index of the running peak at the end


def returns(equity: np.ndarray) -> np.ndarray:
    e = np.asarray(equity, dtype=float)
    return e[1:] / e[:-1] - 1 if len(e) > 1 else np.array([])


def drawdown(equity: np.ndarray) -> Drawdown | None:
    e = np.asarray(equity, dtype=float)
    if len(e) < 2 or not np.all(np.isfinite(e)) or np.any(e <= 0):
        return None
    peak = np.maximum.accumulate(e)
    dd = e / peak - 1
    trough = int(np.argmin(dd))
    peak_i = int(np.argmax(e[:trough + 1]))
    return Drawdown(float(dd[-1]), float(dd[trough]), peak_i, trough, int(np.argmax(e)))


def volatility(rets: np.ndarray) -> float | None:
    r = np.asarray(rets, dtype=float)
    return float(r.std(ddof=1) * math.sqrt(SESSIONS_PER_YEAR)) if len(r) > 1 else None


def downside_volatility(rets: np.ndarray, threshold: float = 0.0) -> float | None:
    """Annualized root mean square of the shortfall below `threshold`, over all sessions (a session above the
    threshold counts as 0). The denominator is n - 1, as for the volatility."""
    r = np.asarray(rets, dtype=float)
    if len(r) < 2:
        return None
    short = np.minimum(r - threshold, 0.0)
    return float(math.sqrt((short ** 2).sum() / (len(r) - 1)) * math.sqrt(SESSIONS_PER_YEAR))


@dataclass(frozen=True)
class TailRisk:
    confidence: float
    var: float | None               # loss (positive) not exceeded on `confidence` of sessions; None if too short
    es: float | None                # mean loss of the sessions in the tail
    n: int                          # returns used
    tail: int                       # sessions in the tail (the k worst)
    min_obs: int


def historical_var_es(rets: np.ndarray, confidence: float, min_obs: int | None = None) -> TailRisk:
    """Historical VaR and Expected Shortfall. The tail is the k = ceil(n x (1 - confidence)) worst returns; VaR
    is the loss of the k-th worst, ES the mean loss of the k worst. Below `min_obs` returns both are None."""
    r = np.sort(np.asarray(rets, dtype=float))
    need = MIN_OBS_VAR.get(confidence, 250) if min_obs is None else min_obs
    n = len(r)
    k = max(1, math.ceil(round(n * (1 - confidence), 9))) if n else 0
    if n < need or n == 0:
        return TailRisk(confidence, None, None, n, k, need)
    return TailRisk(confidence, float(-r[k - 1]), float(-r[:k].mean()), n, k, need)


def hhi(weights: np.ndarray) -> float | None:
    """Herfindahl index of position weights (as fractions of the invested value); 1 / HHI is the effective N."""
    w = np.asarray(weights, dtype=float)
    if len(w) == 0 or w.sum() <= 0:
        return None
    w = w / w.sum()
    return float((w ** 2).sum())

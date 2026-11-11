"""Statistics for pattern research.

The unit of inference is the DATE: events on the same date are averaged first, because a market-wide
move triggers many correlated events at once. Overlapping multi-day outcomes are handled with
Newey-West (Bartlett) standard errors on the date series, lag = horizon - 1.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


@dataclass(frozen=True)
class Summary:
    n_events: int
    n_dates: int
    mean: float | None           # equal weight per date (the tested quantity)
    event_mean: float | None     # equal weight per event (descriptive)
    median: float | None         # event level
    win_rate: float | None       # event level, share of outcomes > 0
    std: float | None            # event level
    se: float | None             # Newey-West SE of `mean`
    t: float | None
    p: float | None
    ci_low: float | None
    ci_high: float | None
    mean_after_cost: float | None


def date_means(dates: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(unique sorted dates, mean value per date, events per date)."""
    order = np.argsort(dates, kind="stable")
    d, v = dates[order], values[order]
    uniq, start, counts = np.unique(d, return_index=True, return_counts=True)
    sums = np.add.reduceat(v, start) if len(v) else np.array([])
    return uniq, sums / counts, counts


def newey_west_se(x: np.ndarray, lag: int) -> float:
    """HAC standard error of the mean of x (Bartlett kernel, no small-sample correction)."""
    n = len(x)
    if n < 2:
        return math.nan
    u = x - x.mean()
    s = u @ u / n
    for k in range(1, min(lag, n - 1) + 1):
        weight = 1 - k / (lag + 1)
        s += 2 * weight * (u[k:] @ u[:-k]) / n
    return math.sqrt(max(s, 0.0) / n)


def mean_test(series: np.ndarray, lag: int) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """(se, t, p, ci_low, ci_high) for H0: mean(series) = 0."""
    n = len(series)
    if n < 3:
        return None, None, None, None, None
    se = newey_west_se(series, lag)
    if not se or math.isnan(se):
        return None, None, None, None, None
    mean = float(series.mean())
    t = mean / se
    df = n - 1
    p = float(2 * sps.t.sf(abs(t), df))
    half = float(sps.t.ppf(0.975, df)) * se
    return se, t, p, mean - half, mean + half


def summarize(dates: np.ndarray, values: np.ndarray, horizon: int, cost: float) -> Summary:
    mask = np.isfinite(values)
    dates, values = dates[mask], values[mask]
    if len(values) == 0:
        return Summary(0, 0, *([None] * 11))
    _, series, _ = date_means(dates, values)
    se, t, p, lo, hi = mean_test(series, max(horizon - 1, 0))
    mean = float(series.mean())
    return Summary(
        n_events=int(len(values)),
        n_dates=int(len(series)),
        mean=mean,
        event_mean=float(values.mean()),
        median=float(np.median(values)),
        win_rate=float((values > 0).mean()),
        std=float(values.std(ddof=1)) if len(values) > 1 else None,
        se=se, t=t, p=p, ci_low=lo, ci_high=hi,
        mean_after_cost=mean - cost,
    )


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """BH-adjusted q-values, same order as the input."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    q = [0.0] * m
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, p_values[i] * m / rank)
        q[i] = running
    return q


def benjamini_yekutieli(p_values: list[float]) -> list[float]:
    """BY-adjusted q-values, same order as the input: BH times c(m) = 1 + 1/2 + ... + 1/m, capped at 1.
    Valid under any dependence between the tests (BH needs independence or positive dependence)."""
    m = len(p_values)
    c_m = sum(1.0 / i for i in range(1, m + 1))
    return [min(1.0, q * c_m) for q in benjamini_hochberg(p_values)]

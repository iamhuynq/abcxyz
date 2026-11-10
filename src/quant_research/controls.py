"""Random-control summaries shared by the economic and portfolio evaluations (review of 2026-11-10, fix 5).

Every control's outcome is stored (`control_outcomes`), not only a share: the share of controls beaten, the
percentiles, and the empirical exceedance p = (1 + #controls ending at or above the strategy) / (1 + n),
the usual Monte Carlo p-value with the +1 correction (never 0).
"""

import numpy as np

N_CONTROL = 200

SCHEMA = """
CREATE TABLE IF NOT EXISTS control_outcomes (
    run_id VARCHAR NOT NULL, capital DOUBLE NOT NULL, cost_model VARCHAR NOT NULL, series VARCHAR NOT NULL,
    seed INTEGER NOT NULL, final_equity DOUBLE, cagr DOUBLE,
    PRIMARY KEY (run_id, capital, cost_model, series, seed));
"""


def exceedance_p(strategy_final: float, control_finals) -> float | None:
    """Over the controls with a finite final equity; None when there is none (or no finite strategy)."""
    finals = np.asarray(control_finals, dtype=float)
    finals = finals[np.isfinite(finals)]
    if not len(finals) or not np.isfinite(strategy_final):
        return None
    return float((1 + np.sum(finals >= strategy_final)) / (1 + len(finals)))


def summary(strategy_final: float, control_finals, control_cagrs) -> dict[str, float | None]:
    """Share beaten and exceedance p use the same controls: those with a finite final equity. A control
    without one is counted in n_invalid, never as beaten or as beating. Percentiles use the finite CAGRs;
    every statistic is None when nothing is left to summarize."""
    finals = np.asarray(control_finals, dtype=float)
    cagr = np.asarray(control_cagrs, dtype=float)
    valid = finals[np.isfinite(finals)]
    cagr = cagr[np.isfinite(cagr)]
    ok = len(valid) > 0 and np.isfinite(strategy_final)
    return {"share_beaten": float(np.mean(valid < strategy_final)) if ok else None,
            "exceedance_p": exceedance_p(strategy_final, valid),
            "median_cagr": float(np.median(cagr)) if len(cagr) else None,
            "p05_cagr": float(np.percentile(cagr, 5)) if len(cagr) else None,
            "p95_cagr": float(np.percentile(cagr, 95)) if len(cagr) else None,
            "n_controls": float(len(valid)),
            "n_invalid": float(len(finals) - len(valid))}


def outcome_rows(run_id: str, capital: float, cost_model: str, series: str, finals, cagrs) -> list[tuple]:
    return [(run_id, capital, cost_model, series, seed, None if not np.isfinite(f) else float(f),
             None if not np.isfinite(c) else float(c))
            for seed, (f, c) in enumerate(zip(finals, cagrs))]

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


def exceedance_p(strategy_final: float, control_finals) -> float:
    finals = np.asarray(control_finals, dtype=float)
    return float((1 + np.sum(finals >= strategy_final)) / (1 + len(finals)))


def summary(strategy_final: float, control_finals, control_cagrs) -> dict[str, float]:
    finals = np.asarray(control_finals, dtype=float)
    cagr = np.asarray(control_cagrs, dtype=float)
    return {"share_beaten": float(np.mean(finals < strategy_final)),
            "exceedance_p": exceedance_p(strategy_final, finals),
            "median_cagr": float(np.nanmedian(cagr)),
            "p05_cagr": float(np.nanpercentile(cagr, 5)),
            "p95_cagr": float(np.nanpercentile(cagr, 95)),
            "n_controls": float(len(finals))}


def outcome_rows(run_id: str, capital: float, cost_model: str, series: str, finals, cagrs) -> list[tuple]:
    return [(run_id, capital, cost_model, series, seed, float(f), None if not np.isfinite(c) else float(c))
            for seed, (f, c) in enumerate(zip(finals, cagrs))]

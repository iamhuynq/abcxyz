# Risk Engine (plan, 2026-11-11)

Source: the external spec "Risk Engine - Implementation Plan" (roadmap step 8, 2026-11-11). This plan holds
two things:
- the Phase A source audit, which describes current behaviour;
- the decisions the spec asks for before any implementation (its section 16).

The Risk Engine measures, explains and checks portfolio risk against declared limits. It never changes weights,
trades, research results or registry states.

## 1. Phase A audit: what exists

| Component | What it does today | Use for risk | Gap or problem |
|---|---|---|---|
| `exposure.py` (`compute`) | Pure function. It takes daily returns (sessions x symbols, NaN = no clean return), the market return and weights. It returns per-stock beta, annualized volatility and risk share; portfolio beta, volatility, effective bets (eigenvalue participation ratio) and effective N of weights (1 / sum w^2); group weights; factor tilts with coverage. | Reuse as is for beta, volatility, risk contribution and effective bets. | **A symbol with fewer than 80% of sessions is dropped and the remaining weights are renormalized.** It is listed in `missing`, but the portfolio numbers then describe a different portfolio. A group label that is not found becomes `"unknown"`. The risk layer must turn both into explicit data-quality conditions (INCOMPLETE above a declared missing weight); it must not pass them. |
| `stock_ui/exposure_data.py` | Loads 120 sessions of clean returns (`daily_panel`; traded, no price jump, not the bad source date), the market return, factor z on the latest date, and **clusters of the latest month**. | The return rules are reused. | The loader lives in the UI package, so the CLI cannot use it. It moves to `quant_research` and the UI calls it. Clusters are the latest month, not the month of the portfolio date. |
| `backtest/metrics.py` | `drawdown` (max drawdown, longest duration), `curve_metrics` (volatility annualized x sqrt(250), ddof 1), `participation` (order / ADV20 with a valid-ADV filter), `WRITEDOWN_SESSIONS = 60`. | Reuse drawdown, volatility and the valid-ADV filter. | No current drawdown with dates, no downside volatility, no VaR / ES. |
| `backtest/costs.py` | Cost model v1: commission and tax, plus the half-spread (max of the tick floor and CHL) and square-root impact `k x sigma x sqrt(notional / ADV)`, with fallbacks when inputs are missing. Inputs come from `stock_trading_costs` on the previous close. | The liquidation cost estimate uses `CostModel.side` unchanged. | Fallback costs are used silently, except for a counter. The risk report must show them. |
| `portfolio/engine.py`, `portfolio/evaluate.py` | Simulate the construction grid. They store **summary metrics only** (`portfolio_results`: CAGR, volatility, max drawdown, participation, industry weights, stale share, ...). | Possible later. | No daily equity curve and no positions are stored, so a stored portfolio run cannot be risk-evaluated without re-simulating it. |
| `backtest/runner.py` | Event-engine backtests. They store `backtest_daily` (equity, cash, invested, n_positions) and `backtest_trades` (entry and exit dates and prices, shares). | Holdings at any date can be rebuilt from the trades, and the equity curve is stored. | - |
| Paper portfolio (`daily.py`) | Stores `paper_daily` (equity with the pre-registered marks, `equity_traded_mark`), `paper_positions` per re-simulation (`run_date`, symbol, shares, `last_price`) and `paper_state`. | The current holdings snapshot is the main monitoring target (20 names, about 1.02 bn VND on 2026-11-09). | Only 26 return observations so far, too few for VaR / ES. There is no write-down series; it can be computed from the positions' last traded dates. |
| `ResultsStore` and provenance | Run lifecycle (running -> ok / failed / invalid), provenance (commit, data, code, config, env, selection), additive `CREATE TABLE IF NOT EXISTS` schema. | A risk evaluation is a run of kind `risk`. | Risk runs log no hypothesis, so they never enter the FDR family. |
| Cross-stock `clusters_monthly` | Correlation clusters per month end from 2006-12 to 2026-11 (240 months). | A point-in-time cluster for any portfolio date. | - |

Reference values from the research period only (VNINDEX up to 2023-12-31), for the stress magnitudes:

| Horizon | 1% quantile | Worst |
|---|---|---|
| 1 day | -4.4% | -7.4% |
| 5 days | -11.0% | -22.2% |
| 20 days | -24.5% | -41.1% |

There were 29 days with a fall of 5% or more.

## 2. Decisions (spec section 16): recommendations

1. **Return series.** Net equity returns, which include costs: these are the returns that were actually
   experienced. The paper portfolio stores only net equity, and backtests store net equity. A gross view is
   not added in v1.
2. **Valuation.** Follow the 2026-10-09 rule. Compute every equity-based metric under the default marks and
   under the 60-session write-down of stuck positions, report both, and judge limits on the worse one. For the
   paper portfolio there are two marks:
   - the pre-registered marks, plus the traded marks (`equity_traded_mark`);
   - a write-down built from each position's sessions without a trade. It is used for current-state metrics
     only, because the paper portfolio has no historical write-down series.
3. **VaR horizon.** Daily only in v1. A 5- or 10-session view would need overlapping windows or a scaling
   assumption, so it is deferred.
4. **Confidence.** 95% and 99%. Historical method: VaR is the empirical loss quantile, ES is the mean loss
   beyond it, and both report the sample size and the tail count.
5. **Minimum observations.** 250 daily returns for the 95% level (about 12 tail points) and 500 for the 99%
   level (5 tail points). Below that, the metric is reported as not available and every limit on it is
   INCOMPLETE. The paper portfolio (26 returns) will be INCOMPLETE for VaR / ES, which is honest.
6. **Policy scope.** One named policy `default`, version `v1`, marked **provisional**. It is applied to both
   research portfolios and the paper portfolio. A policy is a frozen record: a change creates a new version.
   Separate research and monitoring policies come only if a need appears.
7. **Clusters.** Point in time: the cluster assignment of the last month end at or before the valuation date.
   The UI panel switches to the same rule.
8. **Factor limits.** None in v1. Factor tilts are displayed, with their coverage. The only governed
   sensitivity is the absolute portfolio beta.
9. **Stress magnitudes.** Market shocks of -5%, -10% and -15%:
   - -5% is beyond the research-period 1-day 1% quantile (-4.4%);
   - -10% is about the 5-day 1% quantile (-11.0%);
   - -15% lies between the 5-day and 20-day tails.

   Industry and cluster shocks: -10% and -20% on one group at a time, on top of nothing. They are not added to
   the market shock, which avoids double counting. Liquidity shock: ADV halved, with impact recomputed through
   the cost model. Suspension: the largest holdings cannot trade for N sessions and are valued at the
   write-down.

   All of these are illustrations, and the report says so. No magnitude is chosen from holdout or forward
   outcomes, nor from any portfolio's performance.
10. **Persistence.** A risk evaluation is a `research_runs` row of kind `risk`. That gives the lifecycle and
    provenance for free, and it logs no hypothesis. New tables keyed by `run_id`:
    - `risk_policies` (policy_id, version, config JSON, created_at; insert only);
    - `risk_metrics`;
    - `risk_limit_results`;
    - `risk_positions`;
    - `risk_stress_results`.

    The run's parameters name the evaluated portfolio (paper `run_date`, or backtest `run_id` with a date), the
    policy version and the risk model version. Nothing existing is altered.

## 3. Proposed v1 policy `default/v1` (provisional; not tuned on any outcome)

| Limit | Warning | Breach | Rationale |
|---|---|---|---|
| Max position weight | 10% | 15% | The frozen strategy holds 20 names at 5% each |
| Max industry (ICB level 2) weight | 30% | 40% | The construction grid caps industries at 30% at rebalance; actual weights reached 36% to 44% |
| Max cluster weight | 30% | 40% | Same logic for hidden concentration |
| Absolute beta | 1.2 | 1.5 | Long-only equity book; a beta above 1.5 is a leveraged market bet |
| Min effective bets | 5 | 3 | Diversification floor |
| Illiquid weight (position > 1 day of 10% ADV participation) | 20% | 40% | The frozen strategy hits the 5% ADV cap from 10 bn VND |
| Stale-price weight (no trade for more than 5 sessions) | 5% | 15% | `STALE_SESSIONS = 5` in the engines |
| Max drawdown (worse valuation) | -20% | -30% | Monitoring level |
| Missing-data weight (no returns, no ADV or no group) | 0% | - | Above 0%, the limits that depend on that data are INCOMPLETE, never PASS |

Comparisons are inclusive at the threshold (`>=` is a warning or a breach), and this is tested at, just
below and just above each threshold.

The overall status takes the most severe state: INCOMPLETE > BREACH > WARNING > PASS. INCOMPLETE ranks first
because a result without its inputs cannot be trusted. Each status lists every reason, with the affected
positions or groups.

## 4. Scope and PRs

The spec has seven phases. Recommended split:

- **PR 1 (this plan, once approved):** A (this audit), B (metrics), C (policy), D (liquidity), F (persistence,
  CLI, report). Targets: the paper portfolio's latest snapshot, and stored backtest runs at a chosen date.
  - Commands: `quant risk evaluate --paper` or `--backtest RUN [--date D]`, `--policy default/v1`;
    `quant risk policies`; `quant risk report RUN`.
  - The exposure loader moves from the UI to `quant_research`, with no behaviour change; the UI tests guard
    this.
- **PR 2:** E (stress) and G (a read-only UI panel). Also the construction grid: its runs store no curve or
  positions, so it needs a deterministic re-simulation from the stored configuration, or storing daily equity
  for new runs. That is decided in PR 2.

## 5. Tests (PR 1)

- Hand-computed fixtures:
  - drawdown with recovery and dates;
  - volatility and downside volatility;
  - VaR / ES on a 20-value vector (tail off-by-one);
  - beta on an exact linear series;
  - risk shares summing to 1;
  - HHI and effective N on equal and concentrated weights.
- Policy:
  - every limit at, below and above its threshold;
  - INCOMPLETE when returns, ADV or a group label is missing;
  - the severity order;
  - a new policy version leaves earlier evaluations unchanged.
- Liquidity:
  - ADV valid, zero, negative, NaN and infinite;
  - a suspended position;
  - a stale-marked position.

  Invalid ADV is never treated as infinite liquidity or zero cost.
- Integration:
  - evaluate a synthetic paper snapshot and a synthetic backtest run through the CLI;
  - positions, trades, results and the registry are unchanged before and after;
  - the report carries provenance and the policy version.
- Mutation check of every new test (the spec's list: drawdown sign, annualization, VaR off-by-one, dropped
  holdings, infinite ADV, boundary operator, missing exposure as zero).
- CI stays synthetic.

## 6. Out of scope (v1)

Weight changes, rebalancing or orders of any kind. Parametric, EVT or Monte Carlo VaR. Factor limits. Using
holdout or forward outcomes to set limits. Changing a pre-registered rule of the paper portfolio. Risk results
create no FDR tests.

## 7. Rollback

New package `quant_research/risk/`, new tables created with `CREATE TABLE IF NOT EXISTS`, and runs of a new
kind. Rollback: revert the merge. The new tables can stay or be dropped; nothing else reads them. Before the
first real evaluation, a backup `data/results.before-risk.duckdb` is taken.

## 8. Success criteria (PR 1)

- The audit gaps in section 1 are closed or stated: the missing-symbol renormalization, the UI-only loader and
  the latest-month clusters.
- A paper-snapshot evaluation on the real store gives a status with every reason. VaR / ES are INCOMPLETE until
  250 returns exist.
- A backtest-run evaluation reproduces the stored max drawdown of that run.
- No existing table changes, apart from new `risk` runs.
- The full suite is green, and every new test fails under its mutation.

## Result, PR 1 (2026-11-11)

Branch `risk-engine`. Full suite: 520 passed (483 before). All plan decisions were applied as written, with one
addition: risk evaluations are refused on validation and holdout dates. Those periods are frozen, and a risk
run on them would also add to the holdout run count.

### Code

- `quant_research/risk/`:
  - `metrics.py`: drawdown with indices, volatility, downside volatility, historical VaR / ES with their sample
    and tail counts, HHI;
  - `liquidity.py`: horizon at 10% of ADV20, cost with fee, tax and cost model v1, stale and floor flags;
  - `limits.py`: frozen versioned policies, inclusive thresholds, INCOMPLETE first;
  - `data.py`: paper and backtest snapshots, market inputs, point-in-time clusters, factor z;
  - `evaluate.py`: `assess` (pure) and persistence;
  - `report.py`, `cli.py`.
- `quant risk evaluate --paper | --backtest RUN [--date D] [--policy default/v1]`, `quant risk policies`,
  `quant risk report RUN`.
- New tables, all additive: `risk_policies`, `risk_evaluations`, `risk_metrics`, `risk_limit_results`,
  `risk_positions`. A risk run is a `research_runs` row of kind `risk` and logs no hypothesis.
- Audit gaps closed:
  - The Exposure Engine's renormalization over the positions with returns is now visible. Beta and volatility
    are scaled to the NAV by the covered weight, and any position without enough returns makes beta and
    effective bets INCOMPLETE.
  - The exposure loader and `current_paper_date` moved from the UI to `quant_research`
    (`risk/data.exposure_returns`, `paper_state`). On the real data the moved loader gives identical frames
    (25 symbols x 120 sessions).
  - Clusters are point in time.

### Real store

Backup `data/results.before-risk.duckdb`. After the runs, the only changes are the 5 new tables and 2 `risk`
runs in `research_runs`; there are no hypothesis rows.

Paper portfolio at 2026-11-09: **INCOMPLETE**.
- Concentration and liquidity pass:
  - position 7.1%;
  - industry 20.0%;
  - cluster 29.2% where known;
  - illiquid 0%;
  - stale 0%;
  - worst drawdown -4.7%;
  - liquidation cost 1.05% of NAV.
- Beta 0.60 and effective bets 9.7 are INCOMPLETE because CAR (7.1% of NAV) has fewer than 96 clean returns
  in the 120-session window.
- The cluster limit is INCOMPLETE: 7 of the 20 holdings (BMI, CAR, CRE, FMC, G36, MBG, TTF) are outside the
  cross-stock cluster universe (top 200 by liquidity), so they have no cluster.
- VaR / ES are not available (25 returns).

Backtest `20261004T184901-backtest-oi_d10_h10_k20_renew-research` at 2023-12-29:
- the stored equity's max drawdown is reproduced (-27.04%, the stored metric is -0.27044);
- AMD, HBB and KTB had stopped trading before the date, so they are INCOMPLETE for liquidity and written down;
- the positions rebuilt from the trades differ from the stored equity by 0.32%, which is reported in the note.

A holdout backtest is refused.

### Tests and mutation check

Tests: `test_risk.py` (27, hand-computed) and `test_risk_evaluate.py` (12; synthetic store, CLI, frozen
periods, frozen policies, no change to existing tables).

Mutation check: 22 mutations covering the spec's list and more, all caught by an assertion.
- The first pass was wrong: a same-size mutation (two digits swapped) left stale bytecode behind. It made
  every later mutation look caught and hid two surviving ones (beta not scaled to NAV; a position counted as
  held on its exit day). Both got tests.
- The loop now runs with `PYTHONDONTWRITEBYTECODE=1`.
- Earlier mutation checks in this project mutated SQL (read at run time) or changed the file size, so they
  were not affected.

### Open decision

The cluster limit and stocks outside the cluster universe: see the reply of 2026-11-11.

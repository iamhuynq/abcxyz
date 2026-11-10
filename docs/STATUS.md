# Project status

Last updated: 2026-11-10. Update this file at the end of every task.

## Data

- Warehouse, research and cross databases through session 2026-11-09 (`data/.daily_done` = 2026-11-09),
  updated 2026-11-10. Collection is manual: the user asks, or runs `scripts/daily.sh` then
  `quant cross build`. No scheduler (confirmed again 2026-10-24).
- Paper portfolio (frozen strategy, forward data) since 2026-10-03. At 2026-11-09, after 26 sessions:
  - strategy +1.86%, above the p95 of the random controls (-0.08%; median -2.23%);
  - equal weight -1.84%, VNINDEX -2.22%;
  - before realistic costs, and too short a sample to conclude anything.
- Avoid signal hits on forward data: KOS (2026-10-27, 2026-10-29), HIO (2026-10-29), POM (2026-10-30).

## Roadmap (design doc phases)

| Phase | State | Docs |
|-------|-------|------|
| 1 Historical database | Built (crawler, warehouse, validation, daily catch-up) | `crawler-plan.md`, `daily-pipeline-plan.md`, `data-dictionary.md` |
| 2 Feature engine | Built | `feature-engine-plan.md`, `feature-dictionary.md` |
| 3 Pattern discovery | Built and validated | `pattern-discovery-plan.md`, `validation-results-2026-10-04.md` |
| 4 Cross-stock | Built; 4b industry groups built | `cross-stock-*.md`, `group-analysis-*.md`, `group-momentum-validation-2026-10-08.md` |
| 5 Event engine | Built (catalog v1, 12 types, event study) | `event-engine-plan.md` |
| 6 Backtesting | Built; holdout failed; forward paper trading running | `backtest-plan.md`, `backtest-results-2026-10-04.md` |
| 7 Machine learning | Not started | - |

Also built: Streamlit UI (`ui-plan.md`), Market watch dashboard (`dashboard-plan.md`), run provenance and
Exposure Engine (`provenance-exposure-plan.md`), Research Registry (`research-registry-plan.md`),
Factor Engine f1 (`factor-engine-plan.md`), Regime Engine and Interaction Engine
(`regime-interaction-plan.md`), Economic Validation Layer (`economic-validation-plan.md`), Portfolio
Construction (`portfolio-construction-plan.md`).

Stopped: fundamentals (direction B), `fundamental-plan.md`, stopped by the user on 2026-10-08.

## Research conclusions

Source of truth for decisions: the Research Registry (`uv run quant registry list`, Research page >
Registry). Summary:

- No signal tested so far is profitable after trading costs.
- Only validated signal: the avoid signal `drop3_volume2_sellers` (PASS).
- Order imbalance is predictive but uneconomic after costs.
- Portfolio backtest failed on the holdout. Economic evaluation (2026-10-09): the in-sample edge was real
  selection skill (it beats random, industry-matched and beta-matched controls), but it absorbs only
  0.415% of extra cost per side, while realistic spread and impact cost 0.6% to 0.8%. Its CAGR is
  -10.9% under cost model v1, against +14.3% with flat costs (`economic-results-2026-10-09.md`).
- Portfolio construction grid (2026-10-09, six configurations): none beats equal weight after realistic
  costs.
  - Slower order-flow portfolios cut turnover but lose the signal.
  - Top-20 momentum 12-1 underperforms even before costs: extreme winners lag, and it is worse than
    random portfolios with the same turnover.
  - Order flow beats turnover-matched random portfolios, but cannot pay its costs.
  - Industry caps hold at rebalance only: actual weights reach 36% to 44% between rebalances. T+2 is
    immaterial (PR #1 review fixes).
  - The Bear filter helps, but not enough. `P7-IX-momentum_12_1-direction` is rejected
    (`portfolio-results-2026-10-09.md`).
- Cross-stock lead-lag lives in the opening gap (stale prices), not in tradable returns.
- Industry momentum failed validation (2026-10-08); the hypothesis is closed.
- Event study (descriptive): downside events are followed by underperformance; BREAKOUT is the only new
  positive candidate, tracked on forward data.
- Interaction scan (research period, 2026-10-08): 14 of 54 factor x regime tests pass the declared
  candidate rule (registry `P7-IX-*`). They form about 3 to 4 themes: order flow and liquidity matter more
  in weak / risk-off markets, and momentum 12-1 works in Bull, not Bear. Most effects fade after 2015.
  These are statistical only; confirmation needs a pre-registration and forward data
  (`interaction-results-2026-10-08.md`).

## Open items

- PR #1 (`research-platform`) was merged into `main` on 2026-10-09 (`511bb2b`), and CI is green on
  `main`.
  `CLAUDE.md` stays local: it is ignored by the user's global gitignore.
- Whitelist tightened (2026-10-09). Tickers are upper-case letters and digits with at most one upper-case
  hyphen suffix; ICB codes are digits only. `/symbols/all-financial-data` and other lower-case endpoint
  names are rejected. All 2,012 warehouse symbols and every ICB code still pass.
- Execution and data audit (external review of 2026-10-09; plan `execution-audit-plan.md`, branch
  `execution-audit`; merged as PR #2).
  - **Phase A done:** per-lot T+2, invalid ADV blocked in both engines, marks at last traded closes
    (the paper portfolio keeps its pre-registered marks and adds a traded-mark line), invariant tests,
    and error-level validation that stops `daily.sh` (exit 8). The re-runs change no verdict.
  - **New finding:** positions in stocks that stop trading stay at their last price and flatter results.
    To be decided in phase B.
  - **Phase B done:** `quant audit pit`.
    - Delisted stocks are covered (459). 173 listed stocks have been silent for more than 30 days
      (suspensions).
    - Limit flags are band-sensitive on 2% to 8% of liquid rows.
    - A write-down of stuck positions lowers every CAGR, by 0.5 to 7 points; no verdict changes.
    - Pending decision: the valuation policy for future evaluations (recommended: report both, and
      judge on the worse).
  - **Phase C done:**
    - valuation policy: report the last price and a 60-session write-down, and judge pre-registered rules
      on the worse;
    - order-size statistics: the frozen strategy hits the 5% ADV cap on every order from 10 bn VND;
    - an end-to-end CLI test on an integration fixture runs in CI;
    - the industry cap is declared rebalance-time.
  - Branch `execution-audit`, PR #2. Its review asked for three fixes: NULL validation, the worse
    valuation in the verdict, and `last_trade` in legacy mode. They are a new commit, and the grid was
    re-run with no verdict change. The user pushes and merges.
- Review of 2026-10-10 (plan `review-2026-10-10-fixes-plan.md`, branch `review-2026-10-10-fixes`, merged as
  PR #3, `f0afc3f`):
  - run lifecycle running / ok / failed;
  - the paper portfolio only uses complete sessions;
  - exact Spearman and tie-break in the decile scan (research scan re-run; no logged test crosses 0.05);
  - consistent stale share;
  - group returns v2 (v1 records unchanged).

  PR #3 review fixes, second commit:
  - v2 redefined as the cumulative relative return versus the market;
  - the paper portfolio stops at the last contiguous complete session;
  - `finish_run` checks the transition, and `quant runs` lists stuck runs.

  Updated PR #3 review fixes, third commit:
  - Market watch v2 checks validity per window;
  - `paper_state` holds the current paper date, never before the forward start;
  - `invalidate` closes the run;
  - fixed a crash of the all-cash paper portfolio.

  Re-review fixes, fourth commit:
  - the 80% rule is rounded up for new definitions (`min_valid`), while legacy definitions keep
    rounding down;
  - `paper_state` is NULL when no paper record was simulated.

  Fifth commit: a missing `paper_state` row falls back to the latest run; an explicit NULL is kept.

- Next from `project-review-vn-stock-quant-1.md` (agreed order 2026-10-08):
  1. Research Registry: done.
  2. Factor Engine: done (f1, 9 factors; no Value / Quality without fundamentals).
  3. Regime / Interaction: done (one research-period scan, 14 candidates).
  4. Economic Validation Layer: done (cost model v1, benchmarks, matched controls, `quant econ evaluate`).
  5. Portfolio Construction: done (grid C1-C6, none passes; momentum x direction rejected).
  6. Remaining review item: ML (last; the review advises against it until a signal survives costs).
     Rule from now on: any strategy candidate passes `quant econ evaluate` / `quant portfolio evaluate`
     (research period) before a pre-registration. The portfolio control is persistence-matched
     (fixed 2026-10-09).
- Backups `data/results.before-registry.duckdb`, `data/results.before-interactions.duckdb`,
  `data/results.before-econ.duckdb`, `data/results.before-portfolio*.duckdb` (v1 to v3) and
  `data/results.before-execution-audit.duckdb` can be deleted once the user is satisfied.

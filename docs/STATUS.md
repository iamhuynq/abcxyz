# Project status

Last updated: 2026-11-11. Update this file at the end of every task.

## Data

- Warehouse, research and cross databases through session 2026-11-09 (`data/.daily_done` = 2026-11-09),
  updated 2026-11-10. Collection is manual: the user asks, or runs `scripts/daily.sh` then
  `quant cross build`. No scheduler (confirmed again 2026-10-24).
- Paper portfolio (frozen strategy, forward data) since 2026-10-03. At 2026-11-09, after 26 sessions:
  - strategy +1.86%, above the p95 of the random controls (-0.08%; median -2.23%);
  - equal weight -1.84%, VNINDEX -2.22%;
  - against 200 random controls (diagnostic, not the pre-registered test) the exceedance p is 0.020;
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
| 7 Machine learning | Started: PR #10 foundation (contracts, splits) | `ml-plan.md` |

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

- Review of 2026-11-10 (plan `review-2026-11-10-fixes-plan.md`, branch `review-2026-11-10-fixes`, merged as
  PR #4, `5865675`; the branch review leaned to approval, and its three points are addressed):
  - the research universe is split into a point-in-time universe and named outcome masks; recorded
    definitions are unchanged;
  - `quant audit masks`: no recorded conclusion depends on the masks
    (`mask-sensitivity-results-2026-11-10.md`);
  - `daily.sh` exits 9 when busy, shown as refused in the UI;
  - raw files are never replaced;
  - 200 stored random controls with an exceedance p for new econ and portfolio runs; the paper portfolio
    keeps its pre-registered 20 and adds a 200-control diagnostic.
  - Backlog: Retry-After as an HTTP date; an `adj_ratio` overlap audit; a backup and restore runbook
    (the BY column and the vendor-schema canary are done, below).
- Post-merge checklist (external, 2026-11-11): checked on `730464a`. 406 tests pass, the raw store never
  exposes partial files, and the mask audit already has provenance. Its P2 items A and B were done (plan
  `contract-and-fdr-plan.md`, branch `contract-and-fdr`, merged as PR #5, `1cf111a`):
  - payload contracts: `fireant validate` checks the latest raw files and stops `daily.sh` on a contract
    error (`--no-contract` skips the check); `fireant contract scan` over all 26,428 raw files finds 0
    errors;
  - BY q-values: `hypothesis_q.q_by` and `quant audit fdr`. Four of the 13 open interaction candidates
    (foreign_flow-breadth, liquidity-liquidity, order_flow-foreign, reversal_1w-liquidity) pass BH but not
    BY. This is descriptive: the registry is unchanged.
  - Not done: cost model and historical limit bands (no source for historical bands yet).
  - CI fix on the branch: the UI task runner now releases the pipeline lock before it marks a run ended
    (a race already on `main`, hit by the slow CI runner).
  - PR #5 review fixes (same branch): contract dates must really parse, the FDR report picks one row per
    hypothesis and names the global family, and a contract-failure runbook is in `README.md`. 451 tests pass.
- PR #5 was merged on 2026-11-11 without its review-fix commit, which had not been pushed. The commit is on
  branch `pr5-review-fixes` (one cherry-picked commit on `main`, 451 tests pass). It goes in its own PR, before
  PR #6.
- Research integrity audit (external spec "PR #6", 2026-11-11; plan `research-integrity-plan.md`, report
  `research-integrity.md`, branch `research-integrity-audit` on top of `pr5-review-fixes`, not pushed):
  - every research table is classified as point in time or outcome. Tests cover future perturbation of every
    column, vendor `adj_ratio` restatements, symbol isolation, incomplete outcome windows and the period
    boundaries;
  - fixed F-1: with per-horizon masks, the pattern lift compared different rows on its two sides. Recorded
    runs (legacy20) are unaffected, and no mask-audit p-value crosses 0.05;
  - fixed G-1: a NaN p-value would have entered the FDR family (the store has none). `quant audit fdr` now
    lists invalid runs and their re-runs: all 39 were re-run;
  - provenance records library versions; a reproducibility test runs the same build and experiment twice.
    Floats agree within 1e-9 (largest difference seen: 2.6e-13);
  - fixed H-2 (owner approved): decile selections break ties by symbol (`patterns.py`, `backtest/data.py`).
    The order_imbalance selections and the frozen strategy's data are identical to before. `momentum10_d10`
    (P3-C2, failed) is now deterministic; it varied by 74 rows between runs;
  - 476 tests pass; 21 mutations, all caught; `results.duckdb` and the registry are unchanged.
  - Pushed as PR #7. Review fixes (ChatGPT review, 2026-11-11), same branch:
    - invalidation policy, and the append-only `run_invalidations` table (backup
      `data/results.before-pr7-fixes.duckdb`);
    - `quant audit fdr` compares the family with all 555 attempts (no BY verdict changes; BH 120 vs 123), shows
      each invalid run's re-run before and after, and has a provenance table. Only 8 of 65 family runs have
      provenance, all made with uncommitted code (H-4);
    - `selection` in provenance;
    - tests for join fan-out, cross-sectional tables and a mid-series gap;
    - report wording qualified; accepted date formats documented;
    - 480 tests pass; 28 mutations in all, every one caught.
  - Second review (head `90cbaf8`), same branch:
    - an invalidated run's successor must be the same experiment, not only the same name. 37 of 39 match;
      2 cross tests were re-run with a redefined test, and the report now says so;
    - an invalid run cannot be invalidated again;
    - the decile tie-break is one constant, shared by both selections and by provenance;
    - 483 tests pass; 34 mutations in all, every one caught.
- PR #7 (research integrity audit) merged into `main` (`da49c49`).
- Risk Engine PR 1 (plan `risk-engine-plan.md`, branch `risk-engine`, PR #8):
  - `quant risk evaluate --paper | --backtest RUN`: limits under the default policy `default/v2` (provisional;
    `default/v1` kept for reproducibility), liquidity,
    drawdown under both valuations, and VaR / ES once 250 returns exist;
  - status PASS / WARNING / BREACH / INCOMPLETE: missing data never passes;
  - additive `risk_*` tables, runs of kind `risk`, no hypotheses logged;
  - paper portfolio at 2026-11-09: INCOMPLETE. Concentration, liquidity and drawdown pass. Beta is unavailable
    because CAR has a short history, and 7 of the 20 holdings are outside the cluster universe;
  - 520 tests pass; 22 mutations, all caught. A stale-bytecode pitfall in the mutation loop was found and fixed;
  - PR 2: stress scenarios, UI panel, construction-grid integration;
  - PR #8 review fixes (same branch):
    - policy `default/v2` splits the historical drawdown from the current write-down drawdown;
    - unclustered weight is reported (paper: 36.2%), and the cluster limit stays INCOMPLETE while any holding
      lacks a cluster;
    - unvalued positions, or a reconciliation gap above 1% of NAV, make the position limits INCOMPLETE;
    - non-finite returns give no VaR;
    - cost fallback weight is reported;
    - float tolerance at the thresholds;
    - 531 tests pass; 32 mutations, all caught.
  - Second review fixes: the paper portfolio's traded-mark curve is required. A curve that cannot be measured
    (missing marks) makes the historical drawdown limit INCOMPLETE instead of being skipped. README and STATUS
    name `default/v2` as the default. 534 tests pass; 34 mutations, all caught.
- PR #8 (Risk Engine PR 1) merged into `main` (`a46c661`).
- Risk Engine PR 2 (plan `risk-engine-pr2-plan.md`, branch `risk-engine-pr2`, PR #9):
  - stress scenarios `stress/v1` (market, industry, cluster, liquidity, suspension) in every risk run
    (model `risk-v2`), MEASURED / INCOMPLETE, never summed;
  - `quant risk evaluate --construction C1..C6`: a labelled current re-simulation, reconciled with the stored
    run. Phase 0 found the historical grid runs not reproducible (warehouse not kept, uncommitted code); none
    of C1 to C6 reconciles, and C1 / C4 have identical equity;
  - read-only UI page "Risk" and a "Risk: evaluate paper portfolio" task;
  - paper at 2026-11-09: uniform market -10% -> -10% of NAV; worst cluster -20% -> -5.9% (INCOMPLETE, 36.2%
    unclustered); ADV halved -> 14.6% illiquid; top-3 suspension past 60 sessions -> -17.3%;
  - 556 tests pass; 22 + 34 mutations, all caught.
  - PR #9 review fixes:
    - the construction snapshot reads the engine's own state (`run_targets(snapshot_at=...)`);
    - reconciliation has declared required and not-recomputed metric sets: a missing metric or a NULL is never
      a match;
    - stress rejects non-finite betas, ADV and values;
    - the suspension write-down is labelled as an assumption;
    - 565 tests pass; 30 + 34 mutations, all caught.
- PR #9 (Risk Engine PR 2) merged into `main` (`c6ff618`).
- ML, Step 9 (plan `ml-plan.md`). PR #10, the ML foundation (branch `ml-foundation`, not pushed):
  - contracts, a feature allowlist with source versions, deterministic SHA-256 hashes (`ml/1`), walk-forward
    folds, interval purge with an embargo gap, and fold checks;
  - the research cutoff 2023-12-31 is enforced on every date and label end;
  - one primary test per ML experiment counts in the global FDR family;
  - no model, no data written, no engine change (only an additive `regimes.version()`);
  - 646 tests pass; 55 mutations, all caught. Three review rounds:
    - first: immutable specs, a pinned baseline, the split covering the dataset period, strict purge;
    - second: a canonical regime version including `TERCILE_DIMENSIONS`; an ML factor source version that
      includes every `FactorParams` field; `check_fold` validating the fold boundaries itself;
    - third: `feature_builds` records the factor and regime parameters actually used, and `ml/builds.py`
      refuses a build that is not under the approved configuration, or whose configuration is unknown. The
      current `research.duckdb` predates this and is refused until the next `quant build`;
  - next: PR #11, the point-in-time dataset builder.
- Branch `fireant-api-inspector` (Chrome extension that found the trade-tape hub methods,
  `fireant-api-inspector-plan.md` on that branch): local only. The user stopped this work on 2026-11-11, and its fate
  (keep or delete) is the user's call.
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

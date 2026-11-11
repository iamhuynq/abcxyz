# Research integrity and reproducibility audit (2026-11-11)

Plan: `research-integrity-plan.md`. Branch `research-integrity-audit`. Scope: the research pipeline as it is
on 2026-11-11 (`main` at `1cf111a` plus the PR #5 review fixes). This document describes current behaviour.

## 1. Summary

The pipeline holds its main invariants, and they are now enforced by tests across every table:
- point in time: data after t never changes a feature, factor, regime, event or cost input at t;
- symbol isolation: one stock's data never changes another stock's own series or outcomes;
- chronological periods with purge masks;
- one global FDR family that can be traced row by row.

The audit found three defects, all Low:
- **F-1 (fixed):** the per-horizon mask mode compared different rows on the two sides of a pattern lift.
- **G-1 (fixed):** a NaN p-value would have entered the FDR family.
- **H-2 (open, owner decision):** decile patterns break ties in an undefined order, so `momentum10_d10` selects
  slightly different rows on each run.

None changes a recorded decision. `results.duckdb` and the registry are unchanged: the same row counts and the
same states before and after the audit.

What cannot be verified with the current data is listed in section 6. It covers:
- historical exchange, industry and index membership;
- publication times;
- vendor restatements before our first fetch.

FDR correction does not repair any of these. Results stay "survivorship-limited and point-in-time only for price
and flow data". They are not survivorship-free.

## 2. Pipeline

| Stage | Entry point | Module | Output | Tests |
|---|---|---|---|---|
| Raw data | `fireant update` / `backfill` (GET-only, whitelist) | `fireant_crawler/jobs`, `store/raw_store.py` | `data/raw/<job>/<key>/<fetch date>/*.json.gz` (never replaced) | `test_jobs.py`, `test_whitelist.py` |
| Normalization, contract | same; `fireant validate`, `fireant contract scan` | `normalize/*`, `validate/contract.py` | `warehouse.duckdb` (`quotes_daily`, `symbols`, `corporate_actions`, ...) | `test_normalize.py`, `test_contract.py`, `test_validate.py` |
| Panel | `quant build` | `sql/01_daily_panel.sql` | `daily_panel` | `test_feature_build.py`, `test_integrity.py` |
| Market and regimes | `quant build` | `sql/02_market_daily.sql`, `regimes.py` | `market_daily`, `market_regimes` | `test_regimes.py`, `test_integrity.py` |
| Features | `quant build` | `sql/03_stock_features.sql` | `stock_features` | `test_feature_build.py` (pure-Python reference), `test_integrity.py` |
| Outcomes | `quant build` | `sql/04_stock_targets.sql` | `stock_targets`, view `feature_target` | same |
| Factors, events, cost inputs | `quant build` | `factors.py`, `sql/06_stock_events.sql`, `backtest/costs.py` | `stock_factors`, `stock_events`, `stock_trading_costs` | `test_factors.py`, `test_events.py`, `test_costs.py`, `test_integrity.py` |
| Cross-stock | `quant cross build` | `cross/*` | `cross.duckdb` | `test_cross.py`, `test_groups.py` |
| Tests of hypotheses | `quant pattern` / `scan` / `cross test` / `interactions scan` / `backtest` / `portfolio evaluate` | `engine.py`, `interactions.py`, `cross/hypotheses.py`, `backtest/runner.py`, `portfolio/evaluate.py` | `results.duckdb`: `research_runs`, `hypothesis_log`, stats tables | `test_engine.py`, `tests/backtest/*` |
| FDR | read time | view `hypothesis_q` (`results.py`) | `q_value` (BH), `q_by` (BY) | `test_stats.py`, `test_fdr_family.py`, `test_fdr_audit.py` |
| Decisions | `quant registry move` | `registry.py` | `hypotheses`, `hypothesis_transitions` | `test_registry.py` |
| Reports | `quant audit pit` / `masks` / `fdr`, result docs | `audit.py`, `mask_audit*.py`, `fdr_audit.py` | `data/reports/research/*.md`, `docs/*-results-*.md` | `test_audit.py`, `test_mask_audit.py` |

Every table of `research.duckdb` is classified in `tests/research/test_integrity.py`:
- point in time: `daily_panel`, `market_daily`, `source_coverage`, `stock_features`, `stock_factors`,
  `market_regimes`, `stock_events`, `stock_trading_costs`;
- outcome: `stock_targets`;
- metadata: `feature_builds`, `factor_definitions`;
- view: `feature_target`.

A new table fails that test until it is classified.

### Identifiers and time

- **Symbol.** The FireAnt ticker, at its current value. There is no stable security id, and ticker changes are
  not tracked (E-1).
- **Trading date.** The session date, Vietnam time. Each symbol's panel starts at its first traded date and
  ends at its last traded date. Within that span, a session without a trade stays as a row with
  `is_traded = false`.
- **Decision time.** The close of session t. Every point-in-time table row for date t uses data up to and
  including session t.
- **Earliest execution.** The open of t+1. Exit is at the close of t+h, with h >= 3 because of T+2 settlement.
  The backtest engine enters at the open from the candidate list built at the previous close
  (`backtest/engine.py` header), and reads costs from the previous close's row.
- **Availability time.** Quotes and flows are known at the close of their session. Report dates and
  fundamentals have no publication timestamp, so they are not used in features (`audit pit` checks that
  `report_marks`, `fundamental_snapshots` and `stock_targets` never appear in feature SQL).

## 3. Methodology invariants

### Look-ahead (workstream B)

- Feature SQL has no `LEAD` and no `FOLLOWING`; target SQL has no `LAG` and no `PRECEDING`. This is a static
  test.
- Perturbation: prices, volumes, order counts and foreign values after a cutoff are changed. Every column of
  every point-in-time table at or before the cutoff stays exactly equal (`test_changing_the_future_*`).
- Vendor restatement: FireAnt's adjusted prices are `price / adj_ratio`. A corporate action makes the vendor
  rescale `adj_ratio` on all earlier rows.
  - Every feature, factor, regime, event, cost input and complete outcome is a ratio, so it is unchanged by such
    a rescale.
  - Only the adjusted levels change (`adj_*`, `mkt_open`, `mkt_close`, `mkt_ma50`, `mkt_ma200`).
  - Rule: adjusted levels are used only in ratios (`test_restated_adjustment_ratios_*`).
- Cross-sectional steps work per date, over that date's universe: factor winsorizing, ranks and z-scores, and
  scan deciles. Regime terciles use a trailing window of 500 sessions with today included.
- Symbol isolation: all windows are `PARTITION BY symbol ORDER BY date`. Changing one symbol leaves every other
  symbol's panel, features, events, cost inputs and outcomes unchanged (`test_one_symbol_never_moves_*`).
  Factors and regimes are cross-sectional by design and are excluded from that test.

### Outcomes (workstream C)

| Column | Formula | Entry | Exit | Window |
|---|---|---|---|---|
| `fwd_ret_close_{1,3,5,10}d` | `adj_close[t+h] / adj_close[t] - 1` | close t (not executable) | close t+h | h sessions of the symbol |
| `fwd_ret_exec_{3,5,10,20}d` | `adj_close[t+h] / adj_open[t+1] - 1` | open t+1 | close t+h | same |
| `fwd_excess_exec_{5,10,20}d` | exec return minus VNINDEX over the same open-to-close span | open t+1 | close t+h | same |
| `fwd_max_return_5d`, `fwd_max_drawdown_5d` | max high / min low over t+1..t+5, relative to the open of t+1 | open t+1 | - | 5 sessions, NULL unless all 5 exist |
| `y_up3_5d` | `fwd_ret_close_5d > 3%` | close t | close t+5 | same |
| `entry_blocked` | open of t+1 at the ceiling | - | - | - |
| `target_complete` | a 20th session after t exists | - | - | - |

- Windows count the symbol's own sessions. A suspension inside a window stretches it in calendar time and is
  not dropped.
- A window that runs past the symbol's last session is NULL, never partial. This is tested at every horizon.
- These are **predictive labels**: gross, with no costs and no capacity limits. Tradable, net-of-cost returns
  exist only in the backtest, econ and portfolio engines (cost model v1, T+2, limit and no-trade blocks, ADV
  cap). Reports keep these two kinds of number apart.

### Splits (workstream D)

- **Fixed chronological periods:**
  - research: up to 2023-12-31;
  - validation: 2024-01-01 to 2025-12-31;
  - holdout: 2026-01-01 to 2026-10-02;
  - forward: from 2026-10-03.
  - The boundary dates are written in seven SQL CASE blocks and four Python constants. A test checks that they
    all agree (`test_period_boundaries_agree_everywhere`).
- **No random splits.** There is no fitted model, and no preprocessing is fitted across periods.
- **Purge, the masks `crosses_period_<h>d`:** a row whose outcome window ends in the next period is dropped
  from research statistics.
- **No embargo is needed.** Features look backward only, so a validation row may read research-period inputs,
  but never research-period outcomes.
- **Gates:** `--prereg` for validation and holdout, `--final` for holdout. Pre-registrations are hashed into
  their runs.
- **Recorded reuse:**
  - Validation was used twice: single-stock patterns and the portfolio backtest on 2026-10-04 (4 pattern and 2
    backtest runs), and the cross family on 2026-10-08 (1 run).
  - Holdout was used once: two backtest runs on 2026-10-04 (`oi_d10_h10_k20_renew` and its `_eq1bn` variant).
    It is frozen and is not described as untouched.
  - Forward is the only unused data.

### Universe and masks (workstreams E, F)

- **Point-in-time universe at t:** traded at t, ADV20 > 1 bn VND, more than 20 sessions of history, no price
  jump at t, and not the corrupt source date 2025-07-16. The ADV uses trailing values only.
- **Named outcome masks** (PR #4): quality (price jump in the window), execution (entry blocked), period
  (window crosses a period). They are applied on top of the universe through `ResearchParams.sample_sql`. Every
  research loader goes through it: patterns, the decile scan, interactions, event study, cross and group tests,
  and the universe baseline of the lift.
- `quant audit masks` (2026-11-10) reports, before and after each mask:
  - row counts;
  - removal by year;
  - concentration inside each pattern.
  No recorded conclusion depends on the masks (`mask-sensitivity-results-2026-11-10.md`).

### FDR (workstream G)

- **Family:** every logged test with a p-value of every run whose status is `ok`. The family is global and
  exploratory. Its membership depends on run status, never on the observed p-value.
  - Pre-registered decisions used their own declared rule.
  - Families are never split after the fact.
- **BH (`q_value`)** is used for decisions. **BY (`q_by`)** is a diagnostic that is valid under any dependence.
- **Traceability.**
  - Each q row comes from exactly one `hypothesis_log` row of one `ok` run, with the same p-value and period.
    On the real store: 295 of 295 rows, with 0 mismatches.
  - Each run stores its provenance.
  - Example: `ix_reversal_1w_foreign` comes from run `20261008T220700-interaction-interaction_scan-research`,
    commit `9b8bdf8`, with p = 0.00232, BH q = 0.00795 and BY q = 0.0498.
- A test whose p-value cannot be computed is kept as NULL. It stays on record, outside the family.
- `quant audit fdr` section 3 lists the tests of invalid runs and the re-run of each one.

### Reproducibility (workstream H)

| Field | Where |
|---|---|
| experiment id | `research_runs.run_id` (`<timestamp>-<kind>-<name>-<period>`) |
| code commit | `params.provenance.git_sha` and `git_dirty` |
| data snapshot | `params.provenance.warehouse_sha` (warehouse file hash at build) and `feature_build_id` |
| config hash | `params.provenance.config_hash`; the full parameters are in `params` |
| strategy version | `pattern_definitions.version` (hash of the condition); `research_code_hash` (all research `.py` and `.sql`) |
| universe id | the universe and mask settings in `params` (`ResearchParams`) |
| date range | `period` (fixed boundaries) |
| random seed | random controls use seeds 0..n-1 by construction; stored per control in `control_outcomes.seed` for runs after 2026-11-10; the clustering seed is fixed in code |
| created at | `research_runs.created_at` |
| artifacts | report paths are printed by each command. Run outputs are tables keyed by `run_id`; there is no artifact table |
| environment | `params.provenance.env`, new: Python, DuckDB, NumPy, pandas, SciPy and statsmodels versions |

**Fixture experiment** (`tests/research/test_reproducibility.py`, `uv run pytest
tests/research/test_reproducibility.py -q`):
- it builds `research.duckdb` twice from one warehouse, and runs a pattern, a base comparison and a decile scan
  twice;
- keys, counts, labels and row sets must be equal;
- floats must agree within a relative 1e-9;
- run ids and timestamps are excluded.

## 4. Findings

```text
ID: F-1
Title: Pattern lift compared different rows on its two sides with per-horizon masks
Severity: Low
Status: Fixed
Evidence: engine.run_pattern loaded pattern rows with sample_sql() (masks over the longest horizon) while
  _lift_vs_universe took the universe with sample_sql(<target>) (masks over the target's horizon). A pattern
  matching every row had a lift of -0.00023 at 5 sessions instead of 0.
Affected components: src/quant_research/engine.py (run_pattern); only ResearchParams(mask_window="horizon"),
  used by the combined_horizon configuration of `quant audit masks`.
Potential impact: Recorded runs use legacy20, where both sides are the same SQL; no recorded result is affected.
  In the 2026-11-10 mask audit, combined_horizon pattern lifts at 5 and 10 sessions move by at most 0.0004 and
  no p-value crosses 0.05 (15 patterns x 2 horizons, measured 2026-11-11). That report is not re-run.
Reproduction or verification steps: the regression test below; the measurement script is described in
  research-integrity-plan.md, Result.
Remediation: with per-horizon masks, the pattern side of each lift is reloaded with that horizon's masks.
  legacy20 keeps the old code path.
Regression test: tests/research/test_outcome_masks.py::test_pattern_lift_compares_the_same_rows_on_both_sides
```

```text
ID: G-1
Title: A NaN p-value would enter the FDR family
Severity: Low
Status: Fixed
Evidence: hypothesis_q filtered `p_value IS NOT NULL`; NaN is not NULL in DuckDB and sorts last. With p-values
  0.001, 0.01, 0.02 and NaN, m was 4 and the q of 0.02 was 0.0267 instead of 0.02. stats.py can return NaN.
Affected components: ResultsStore.log_hypothesis, view hypothesis_q (src/quant_research/results.py).
Potential impact: none so far: the store holds 0 NaN p-values (555 logged tests).
Reproduction or verification steps: tests/research/test_fdr_family.py.
Remediation: log_hypothesis stores NaN as NULL (test on record, outside the family) and refuses a p-value
  outside [0, 1]; the view also drops NaN.
Regression test: test_fdr_family.py (NaN, direct NaN, out-of-range, bounds, re-log, ties, run status, trace).
```

```text
ID: G-2
Title: Invalidation removes tests from the family after their results were seen
Severity: Informational
Status: Accepted limitation (now reported)
Evidence: 41 invalid runs; 39 hold 260 logged tests. Every one has a later `ok` run of the same name; every
  invalidation has a written reason, and all are code or method fixes. One of them changed the definition of
  the primary pattern test (mean versus zero to lift versus universe, 2026-10-04), in the research period and
  before any pre-registration.
Affected components: ResultsStore.invalidate, hypothesis_q.
Potential impact: the family keeps one copy of each test, which is the intended count. A re-run made because a
  first result looked wrong is a fork in the analysis that the family does not price; the recorded reasons and
  the result docs are the control.
Reproduction or verification steps: `uv run quant audit fdr`, section 3 (audit-fdr-20261111T142439.md: 39
  runs, 0 without a re-run).
Remediation: section 3 of the FDR report lists every invalid run with tests and its re-run, and flags runs
  without one.
Regression test: test_fdr_audit.py::test_report_lists_tests_outside_the_family_and_rerun_of_each_invalid_run
```

```text
ID: H-1
Title: Floating-point results are not bit-identical between runs
Severity: Informational
Status: Accepted limitation (bounded and tested)
Evidence: the fixture experiment differs between runs by at most 3.4e-16 (relative) in market_regimes
  liquidity and foreign values, and 2.6e-13 in scan IC statistics. Keys, counts, labels and row sets were equal
  in every one of 3 repetitions.
Affected components: DuckDB parallel aggregation (window sums, per-date corr()).
Potential impact: a value exactly at a threshold could fall on either side; no such case is known.
Reproduction or verification steps: tests/research/test_reproducibility.py.
Remediation: comparison policy: exact for everything but floats, relative 1e-9 for floats.
Regression test: test_reproducibility.py (build twice, experiment twice).
```

```text
ID: H-2
Title: Decile selection breaks ties in an undefined order
Severity: Low
Status: Open (owner decision: adding a tie-break touches a recorded pattern and the frozen strategy's loader)
Evidence: patterns.py `ntile(10) OVER (PARTITION BY date ORDER BY {feature})` and backtest/data.py
  `ntile(10) ... ORDER BY sig` have no tie-break (the decile scan has one since PR #3: `ORDER BY f, symbol`).
  - On the real data, a tie straddles the top-decile boundary on 66 of 5,632 dates for return_10d, and on 0 of
    4,770 dates for order_imbalance.
  - Three identical queries of momentum10_d10 gave row sets that differ by 74 rows; order_imbalance_d10 and
    order_imbalance_d10_no_limit were identical. Industry terciles (cross) have no ties.
Affected components: momentum10_d10 (P3-C2, failed; no forward use). The frozen strategy (order_imbalance) is
  not affected on any date so far, but could be on a future one.
Potential impact: a re-run of momentum10_d10 moves its lift in the fifth decimal (0.00934 versus 0.00933 at 20
  sessions); its decision does not change.
Reproduction or verification steps: run momentum10_d10's source SQL three times and compare the row sets.
Remediation (proposed): add `, symbol` to both ORDER BY clauses. It changes no recorded order_imbalance
  selection (0 ambiguous dates) and makes momentum10_d10 deterministic.
Regression test: to add with the fix (two identical runs give the same rows on a fixture with ties).
```

```text
ID: H-3
Title: Library versions were not part of provenance
Severity: Informational
Status: Fixed
Evidence: provenance.collect recorded commit, data, build, code and config hashes, but no environment.
Affected components: src/quant_research/provenance.py.
Potential impact: a DuckDB or NumPy upgrade could change floats (H-1) or SQL semantics without a trace.
Reproduction or verification steps: the header of any new report shows the versions.
Remediation: `env` key (Python, DuckDB, NumPy, pandas, SciPy, statsmodels) in every new run and report header.
  Older runs keep what they stored.
Regression test: test_reproducibility.py::test_provenance_records_the_environment
```

```text
ID: B-1
Title: Point-in-time coverage of derived tables
Severity: Informational
Status: Verified (tests added)
Evidence: per-table perturbation tests already existed for features, regimes, factors, events, cost inputs,
  cross and group tables, and the backtest. They were missing: one test over every column of every table, a
  vendor-restatement test, and a symbol-isolation test. The plan had over-stated this gap.
Affected components: all of research.duckdb.
Potential impact: none found.
Reproduction or verification steps: tests/research/test_integrity.py.
Remediation: -
Regression test: test_integrity.py; mutation-checked (a lead(), a forward window, an adjusted level, an
  unpartitioned window, a partial outcome window and a shifted period date are each caught).
```

```text
ID: E-1
Title: Exchange, industry and ticker are known only at their current value
Severity: Medium (accepted limitation)
Status: Accepted limitation
Evidence: daily_panel takes exchange_now and the industry codes from the current symbol table. Limit bands and
  tick sizes use the current exchange. There is no index membership or ticker-change history. `quant audit pit`
  (2026-10-09) measured the effect: limit flags are band-sensitive on 2% to 8% of liquid rows; 459 delisted
  stocks are covered; 173 listed stocks have been silent for over 30 days.
Affected components: 01_daily_panel.sql, regimes (breadth), the industry-neutral factor z, industry caps.
Potential impact: a bias whose direction is unknown in industry-relative results and in limit-day handling. It
  is not survivorship in the narrow sense: delisted stocks are in the data.
Reproduction or verification steps: `uv run quant audit pit`.
Remediation: none possible without a historical source. Results are described as survivorship-limited.
Regression test: -
```

```text
ID: E-2
Title: The panel's row range uses each stock's last traded date
Severity: Informational
Status: Accepted limitation
Evidence: daily_panel keeps rows between symbol_trading_span.first_traded_date and last_traded_date. The last
  date is known only later, and it drops the vendor's zero-volume filler after delisting.
Affected components: daily_panel, and the stopped-stock valuation in the backtests.
Potential impact: no feature reads past t. A stock that stops trading leaves the panel at its last trade, so a
  held position keeps its last price. This is covered by the valuation rule of 2026-10-09 (report the last price
  and a 60-session write-down, and judge on the worse).
Reproduction or verification steps: `audit pit` write-down table.
Remediation: -
Regression test: tests/backtest/test_invariants.py::test_writedown_values_stocks_that_stop_trading_at_zero
```

```text
ID: B-2
Title: Event thresholds were set from the research-period distribution
Severity: Informational
Status: Accepted limitation
Evidence: 06_stock_events.sql header: the thresholds are the about-2% tails of each feature in the research
  period, set from features only, never from outcomes.
Affected components: stock_events, the event study, BREAKOUT (P5-E1, monitoring on forward data).
Potential impact: the thresholds are in-sample constants for the research period. They carry no outcome
  information. They are fixed for validation and forward data.
Reproduction or verification steps: -
Remediation: -
Regression test: test_events.py (each event equals its condition; point in time).
```

## 5. Automated coverage

| Test | What it proves |
|---|---|
| `test_integrity.py::test_every_research_table_is_classified` | No table of `research.duckdb` escapes the point-in-time / outcome classification |
| `test_integrity.py::test_changing_the_future_changes_no_point_in_time_row` | Every column of every point-in-time table at or before t is independent of data after t |
| `test_integrity.py::test_restated_adjustment_ratios_change_only_adjusted_levels` | A vendor restatement of `adj_ratio` changes only adjusted levels |
| `test_integrity.py::test_one_symbol_never_moves_another_symbols_series` | No window mixes symbols |
| `test_integrity.py::test_outcome_windows_past_the_last_session_are_null` | Incomplete windows are NULL at every horizon |
| `test_integrity.py::test_period_boundaries_agree_everywhere`, `test_built_periods_follow_the_boundaries` | One set of period boundaries |
| `test_outcome_masks.py::test_pattern_lift_compares_the_same_rows_on_both_sides` | The lift's denominators match in both mask modes (F-1) |
| `test_fdr_family.py` | Family membership and p-value edge cases: NaN, NULL, out of range, re-log, ties, m = 1, run status, trace (G-1) |
| `test_fdr_audit.py::test_report_lists_tests_outside_the_family_*` | Every invalid run is listed with its re-run (G-2) |
| `test_reproducibility.py` | Two builds and two experiments agree under the stated policy; provenance has the environment (H-1, H-3) |
| Existing: `test_feature_build.py`, `test_factors.py`, `test_regimes.py`, `test_events.py`, `test_costs.py`, `test_cross.py`, `test_groups.py`, `tests/backtest/*` | Independent reference implementations and per-table look-ahead tests |

Every new test was mutation-checked: 19 planted defects, all caught by an assertion, not by an error.

## 6. Known limitations

- No historical exchange, industry, index membership or ticker changes (E-1).
- No publication timestamps for reports or fundamentals, so they are not used in features. Fundamentals work
  was stopped on 2026-10-08.
- Adjusted prices come from the vendor. A restatement before our first fetch cannot be seen. A wrong
  `adj_ratio` would distort returns; the `adj_ratio` overlap audit is in the backlog.
- Limit bands and tick sizes are approximate (current exchange; no historical band changes).
- Predictive labels are gross. Tradability depends on cost model v1, which has no history of exchange bands or
  fees.
- The validation period was used twice and the holdout once; only forward data is unused.
- The FDR family is global and exploratory. It cannot price analyses that were never logged, such as reading
  data before a run or choices made in docs, and it does not repair leakage or design errors.

## 7. Follow-up (deferred)

- H-2 tie-break, if the owner accepts it.
- Group C backlog: Retry-After as an HTTP date, the `adj_ratio` overlap audit, a backup and restore runbook.
- An artifact table keyed by `run_id` (report paths), if reports need to be found from a run.
- A source for historical exchange and industry membership, if one appears.

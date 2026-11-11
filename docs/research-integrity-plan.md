# Research integrity and reproducibility audit (plan, 2026-11-11)

Source: the external spec "PR #6 - Research Integrity & Reproducibility Audit" (2026-11-11). It asks for an
audit of the existing pipeline, executable safeguards for the invariants that can be enforced, and one report,
`docs/research-integrity.md`. No new strategies, no methodology change made to improve results.

## Starting point

Much of the spec is already built. The audit consolidates it, checks it against the code, and fills gaps. It
does not redo it.

| Spec workstream | Already in the repo | Gap this plan fills |
|---|---|---|
| A Pipeline inventory | `data-dictionary.md`, `feature-dictionary.md`, plan docs per stage | One current map: stage -> module -> table -> tests, with timestamp semantics |
| B Look-ahead | `03_stock_features.sql` header rule; static test (no LEAD/FOLLOWING in features, no LAG/PRECEDING in targets); `test_perturbing_the_future_does_not_change_past_features` (stock_features only) | The perturbation test does not cover `market_daily`, `market_regimes`, `stock_factors`, `stock_events` or the cross database. No test that one symbol's data never moves another symbol's time-series features |
| C Labels | `04_stock_targets.sql` (close-to-close and executable conventions, `target_complete`, per-horizon masks); reference tests | No written formula/timing table per label; no explicit test that targets never cross symbols, and that incomplete windows are NULL for every horizon |
| D Splits | Fixed chronological periods in `01_daily_panel.sql`; `crosses_period*` purge masks; `--prereg` / `--final` gates; holdout frozen and its reuse recorded | Written split policy; a test that every period gate and purge mask agrees with the period boundaries in one place (the dates are repeated in 6 SQL CASE blocks) |
| E Universe | `quant audit pit` (delisted coverage, stopped stocks, attributes known only at current value, limit-band sensitivity) | State plainly that point-in-time index/exchange/industry membership is not supported; check that report headers say so |
| F Masks | Point-in-time universe split from named outcome masks (PR #4); `quant audit masks` with denominators, by-year and concentration tables | Mask alignment test across pattern, scan and interaction loaders (same rows for the same params) |
| G FDR | `hypothesis_q` (BH, BY), `quant audit fdr`, statsmodels cross-check, tie test | Edge cases: NaN p-values (not filtered by `IS NOT NULL`), duplicate labels, all-NULL; family membership changes through `invalidate` (260 of 555 logged tests sit in invalid runs) |
| H Reproducibility | `provenance.collect`: git SHA + dirty flag, warehouse SHA, feature build, research code hash, config hash; seeds of random controls stored | No library versions (DuckDB, NumPy, pandas, Python) in provenance; no deterministic re-run check |

## Candidate findings (to verify during the audit; severity set only after evidence)

1. **NaN p-values in the FDR family.** `hypothesis_q` filters `p_value IS NOT NULL`, but NaN is not NULL in
   DuckDB and sorts last. A NaN would count in m and shift every q. The store has 0 NaN today, and
   `stats.py` can return NaN. Expected fix: `log_hypothesis` stores NaN as NULL (or refuses it), with a test.
2. **Family membership through invalidation.** Invalidating a run removes its tests from m. All 41 invalid
   runs have a methodological reason, and most were re-run. The audit checks that every invalidated run with
   logged tests has a successor of the same kind and name, and lists any that do not. No historical row is
   changed.
3. **Look-ahead coverage of derived tables.** Extend the perturbation test to regimes, factors, events and the
   cross build. If any changes, it is a defect and is fixed. `stock_events` thresholds came from the
   research-period distribution (documented). They are fixed constants, so no leak at run time, and are
   reported as informational.
4. **Period boundaries repeated in SQL.** `2023-12-31`, `2025-12-31`, `2026-10-02` appear in six CASE blocks
   and in Python. A test asserts that they all agree with one constant. No refactor of the SQL.
5. **Point-in-time universe limits** (exchange and industry at their current value, no index membership,
   back-adjusted prices from the vendor). These are accepted limitations, already measured by `audit pit`.
   They are restated, not re-measured.
6. **Environment in provenance.** Add library versions as a new key. Old runs keep what they stored.

## Work

1. Branch `research-integrity-audit` from `main`.
2. Inventory (A): section "Pipeline" of `docs/research-integrity.md`. For each stage: entry point, module,
   output table, the time each row is known (decision at close t, entry at open t+1, exit at close t+h),
   tests. Current behaviour only.
3. Look-ahead (B): extend the perturbation test to every point-in-time table, plus a cross-symbol isolation
   test. Each new test is mutation-checked: a planted `lead()` or a widened window must fail it.
4. Labels (C): a formula and timing table for every target column. Tests for symbol isolation and incomplete
   windows at every horizon. Name the difference between predictive labels (`fwd_ret_*`) and tradable,
   net-of-cost returns (backtest and econ only).
5. Splits (D): written split policy (chronological, fixed dates, purge by `crosses_period_<h>d`, no fitted
   preprocessing across periods; factor normalization is per date, regime terciles use trailing windows).
   Period-boundary consistency test. Holdout reuse: list the holdout runs from `results.duckdb`.
6. Universe and masks (E, F): restate the `audit pit` and `audit masks` results with their dates. Add a mask
   alignment test across loaders.
7. FDR (G): family definition (global, every logged test of every valid run, exploratory). Fix candidate 1.
   Check candidate 2 with a read-only query. Edge-case tests: NaN, NULL, duplicate label (primary key), ties,
   m = 1. Trace one q-value back to its run and label in the report.
8. Reproducibility (H): add an `env` key to provenance. Add a fixture experiment: build the synthetic research
   database twice, run a pattern and a scan on each, and compare the stored statistics. Floats must be equal
   to 1e-12; run IDs and timestamps are excluded. Document how to run it.
9. Report `docs/research-integrity.md`: summary, pipeline map, invariants, findings in the spec's format,
   coverage (test -> what it proves), limitations, follow-ups.
10. Full suite, README (if a command changes), STATUS, a Result section here, one commit per workstream
    group. The user pushes.

## Not in scope

- No change to recorded definitions, pre-registrations, `registry_seed.py`, stored runs or registry state.
  No re-run of recorded research for new numbers.
- No point-in-time exchange, industry or index membership data: there is no source. It stays a stated
  limitation.
- No ML split machinery: there is no fitted model in the repo (Phase 7 not started).
- Group C items (Retry-After, `adj_ratio` overlap audit, backup runbook): separate milestone.

## Rollback

Code, tests and docs only, on a branch. Revert the merge commit. The provenance `env` key is additive (readers
use `.get`). The NaN guard affects only future `log_hypothesis` calls.

## Success criteria

- `docs/research-integrity.md` exists, with every finding in the spec's format and evidence (file:line, query
  or test).
- New tests pass, and each fails under a planted defect (mutation check recorded here).
- No Critical or High finding left open without the user's explicit acceptance.
- Full suite green; `results.duckdb` and the registry unchanged (same row counts and states, before and after).

## Result (2026-11-11)

Branch `research-integrity-audit`, based on `pr5-review-fixes` (the PR #5 review commit, which missed the
merge; it goes in its own PR first). Report: `research-integrity.md`. Full suite: 476 passed (451 before).
`results.duckdb` and the registry are unchanged (row counts of every table, run states and hypothesis states
compared before and after).

How the candidate findings turned out:

1. NaN p-values: confirmed and fixed (G-1). There are no NaN p-values in the store, so no q-value changes.
2. Invalidation: all 39 invalid runs with tests have a later `ok` re-run. This is now reported in section 3 of
   `quant audit fdr` (G-2), and the real report shows 0 runs without a re-run.
3. Look-ahead coverage: over-stated. Per-table perturbation tests already existed for factors, regimes, events,
   cost inputs and cross. Added instead:
   - one test over every column of every table;
   - a vendor-restatement test;
   - a symbol-isolation test;
   - a table classification test (B-1).
4. Period boundaries: all 11 places agree, and a test now checks them.
5. Universe limits: restated (E-1, E-2).
6. Environment: added to provenance (H-3).

Found during the audit, not in the plan:

- **F-1.** With per-horizon masks, the two sides of a pattern lift used different rows. This was reproduced
  with a test and fixed.
  - Measured on the real research data: 15 patterns x 5 and 10 sessions, using a scratch script that computes
    both lifts with the engine's own `_events` and `universe_by_date`.
  - Largest lift change 0.0004 (`drop3_volume2` at 10 sessions: -0.00648 to -0.00612). No p-value crosses
    0.05.
- **H-1.** Floats are not bit-identical between runs; the largest relative difference is 2.6e-13. A comparison
  policy is written down and tested.
- **H-2.** Decile selections had no tie-break. `momentum10_d10` varied by 74 rows from run to run.
  - The owner approved the fix: `ORDER BY feature, symbol` in `patterns.py` and `backtest/data.py`.
  - The order_imbalance selections and the backtest loader arrays are identical to before.
  - momentum10_d10 is now deterministic.

Mutation check: 21 planted defects, all caught by an assertion:
- 9 look-ahead, label and period;
- 4 FDR family;
- 2 FDR report;
- 3 reproducibility;
- 1 lift masks;
- 2 decile tie-breaks. The first fixture let both tie-break mutations survive, because a small single-thread
  sort happened to pick the right set; the fixture now loads several physical row orders.

`quant audit fdr` was re-run on the real store: `data/reports/research/audit-fdr-20261111T142439.md`.

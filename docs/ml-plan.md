# ML research layer, Step 9 (plan, 2026-11-12)

Source: the external spec "ML Research Layer Plan" (roadmap step 9, 2026-11-12). This document holds the audit
of what the repository already provides, the decisions the spec asks for (its section 11), and the scope of
PR #10 (ML foundation and research contract). PR #11 to #15 get their own plan sections when they start.

## 1. Preconditions

- **The project review's ML policy (section 36) is met.** ML comes after the Factor, Regime and Interaction
  engines, portfolio construction, economic validation and reproducibility. All of them are built.
- **The economic context is unchanged.** No signal tested so far survives realistic costs (STATUS). An ML model
  is one more model family under the same rules: it must beat the Factor Engine baseline out of sample and
  after costs, and its tests count in the multiple-testing family.
- **Research integrity applies as is.**
  - Research period up to 2023-12-31.
  - Validation 2024-2025 (used twice) and the holdout (used once) are frozen.
  - New ideas are confirmed only by a pre-registration on forward data (from 2026-10-03).

## 2. Audit: what ML can reuse

| Need | Existing | Use |
|---|---|---|
| Point-in-time features at the close of t | `stock_features` (30 features), `stock_factors` (9 factors: value, rank, z, industry z), `market_regimes` (6 dimensions) | Feature sources. All three are classified point-in-time in `tests/research/test_integrity.py`, and their perturbation tests cover every column. |
| Labels | `stock_targets`: `fwd_ret_exec_{3,5,10,20}d` (enter at the open of t+1, exit at the close of t+h, the symbol's own sessions), `fwd_excess_exec_{5,10,20}d`, the per-horizon masks `fwd_has_price_jump_{h}d` / `crosses_period_{h}d`, `entry_blocked` | Label sources (outcome table). An incomplete window is NULL (tested). |
| Decision and execution timing | Decision at the close of t, entry at the open of t+1, T+2, exit at a close (`docs/research-integrity.md` section 2) | The same convention, nothing new. |
| Point-in-time universe | `ResearchParams.point_in_time_sql()`: traded, ADV20 > 1 bn, more than 20 sessions, no jump, not the bad source date | The default universe. Exchange and industry are current values (integrity finding E-1): a stated limitation. |
| Periods and gates | `daily_panel.period`, `check_period`, `--prereg` / `--final`, the period-boundary test | ML folds live inside the research period. |
| Runs, provenance, FDR | `research_runs` (lifecycle), `provenance.collect` (commit, data, code, config, env, selection), `hypothesis_log` / `hypothesis_q` (BH, BY), registry | An ML experiment is a run of kind `ml`, and its pre-declared primary test is logged. |
| Not usable as features | Interaction Engine outputs (scan statistics, not per-row values); fundamentals and report dates (no publication time); Exposure and Risk outputs (portfolio-level) | Excluded. Cross-stock features (`cross.duckdb`) are point-in-time tested but live in another database: deferred to PR #11 as an option. |
| Libraries | numpy, pandas, scipy, statsmodels. No scikit-learn and no gradient-boosting library. | Ridge (closed form) and ElasticNet (statsmodels `fit_regularized`) need nothing new. A tree model needs a dependency review in PR #12. |

## 3. Decisions (spec section 11): recommendations

1. **Decision time.** After the close of t, and features must be known at that close. The label is entered at
   the open of t+1. This is the existing convention, everywhere.
2. **Return convention.** The labels are the existing back-adjusted executable returns, gross of costs. Costs
   apply only at the portfolio stage (cost model v1), as in the econ and portfolio engines.
3. **Suspensions and delistings.** A label whose window runs past the symbol's last session is NULL and is
   excluded; it is never filled with a stale price or a zero. Rows whose window holds a price jump, an
   entry-blocked open or a period boundary are excluded through the existing per-horizon masks. The dataset
   reports how many rows each rule removes.
4. **Universe history.** The point-in-time universe above. The current exchange and industry are a stated
   limitation, not fixable without a source.
5. **Feature provenance.** Only columns of tables that the integrity tests classify as point-in-time
   (`stock_features`, `stock_factors`, `market_regimes`). A feature spec naming any other source is refused.
6. **Calendar.** Label intervals use the symbol's own sessions: `label_start` is the first session after t, and
   `label_end` is the h-th. Purge uses the actual interval overlap; embargo is counted in market sessions
   (`market_daily` dates).
7. **Dependencies.** None in PR #10 or #11. PR #12 brings a written dependency review (size, license,
   determinism) before any tree library.
8. **Multiple testing.** Each ML experiment pre-declares one primary test, the incremental rank IC over the
   Factor Engine baseline (Newey-West), and logs it in the global family, like a grid configuration. A sweep
   of configurations logs one test per configuration. The experiment count is visible in `quant audit fdr`.
9. **Holdout governance.**
   - Every fold, including its label intervals, ends on or before 2023-12-31; a split that reaches past it is
     refused.
   - Validation and holdout are not available to ML.
   - A model that passes is confirmed only by a pre-registration on forward data, like every other candidate.

## 4. PR #10 scope: contracts, validation, split primitives

New package `src/quant_research/ml/`. Pure Python, no database writes, no model, no portfolio change.

- `contracts.py`: frozen dataclasses, each with a validator and a stable version hash (as `Pattern.version` /
  `config_version`).
  - `FeatureSpec(name, source, column, availability)`. The source must be one of the point-in-time tables.
    Availability is `close_t` for all three; anything else is refused. A factor feature names the factor and the
    field (`rank_pct`, `z`, `z_industry`).
  - `LabelSpec(column, horizon, entry, exit, costs)`. The column must be in `stock_targets` with a horizon
    suffix that matches. Entry is `open_t+1`, exit `close_t+h`, costs `gross`. It also names the mask rule.
  - `DatasetSpec(features, label, universe, period_start, period_end)`. The period must lie within the
    research period; feature names must be unique. Its fingerprint is a hash of the spec. The data identity
    (warehouse SHA, feature build) is added when built (PR #11).
  - `SplitSpec(kind = expanding | rolling, train_min_sessions, test_sessions, step_sessions, purge = interval,
    embargo_sessions, last_date)`. There is no shuffle field at all. The split is refused when the last test
    window ends after 2023-12-31, or when the embargo is below 0.
  - `ExperimentSpec(hypothesis, dataset, split, model, params, seed, baseline, primary_test)`. It is refused
    without a hypothesis, a baseline (the Factor Engine score named by its factor), a primary test or a seed.
- `splits.py`: pure functions on a session calendar and on label intervals.
  - `folds(calendar, split)`: deterministic fold boundaries.
  - `purge(train_intervals, test_window, embargo)`: drops the training rows whose label interval overlaps
    the test window, or the embargo after it. It works on actual dates, never on row counts.
  - `check_fold(...)`: an assertion helper that later PRs call on every fold. It checks that training precedes
    the test, that no interval overlaps, that the embargo is respected, and that the minimum training size is
    met. It fails clearly rather than weakening a rule.
- `keys.py`: `check_unique(df, ("date", "symbol"))` reports duplicates and refuses them;
  `check_point_in_time(feature_dates, decision_dates)` refuses any feature row dated after its decision date.
- Docs: a "Leakage threats" section in this plan, with every threat from the spec mapped to the rule or test
  that blocks it, and the list of what PR #11 must add (the dataset builder's adversarial tests).

Tests (`tests/research/test_ml_contracts.py`, `test_ml_splits.py`): every validator, with valid and invalid
cases.
- A feature from a table that is not point-in-time, or with availability `close_t+1`, is refused.
- A label with a mismatched horizon is refused, and so is a period past 2023-12-31.
- Duplicate keys are refused.
- Fold boundaries are deterministic and chronological.
- Purge on a hand-made calendar with gaps (a suspension): an interval ending inside the test window or inside
  the embargo is dropped; one ending the day before is kept.
- Too little training data fails.
- Every new test is mutation-checked with `PYTHONDONTWRITEBYTECODE=1`.

Not in PR #10: the dataset builder (PR #11), models (PR #12), the walk-forward runner (PR #13), analysis
(PR #14), portfolio integration (PR #15, conditional). PR #10 changes no existing engine and no stored data.

## 5. Acceptance (PR #10)

- Invalid or ambiguous feature availability, a non-point-in-time source, a label horizon mismatch, a period
  outside research, shuffling (impossible by construction), duplicate keys, purge overlap and a missing
  embargo are all refused by a tested rule.
- The experiment contract carries every provenance field of the spec. The data identity comes from the
  existing `provenance.collect` at run time.
- The full suite is green, CI is synthetic, and no existing engine behaviour changes.

## 6. Plan review (external, 2026-11-12): additions to PR #10

The review accepts the two decisions asked for: fold boundaries and purge in PR #10, and one primary test per
ML experiment in the global FDR family. It adds three requirements, all taken into PR #10.

### 6.1 The research cutoff is an invariant, not documentation

- `RESEARCH_END` is read from `backtest.runner.PERIOD_RANGES["research"][1]` (2023-12-31), the boundary that
  the integrity test already checks everywhere. It is never written again.
- `DatasetSpec` and `SplitSpec` refuse an end date after it.
- `folds()` refuses to produce a fold whose test window ends after it.
- `check_fold()` refuses (raises, never warns) any training or test observation whose decision date **or
  label end** is after it.
- Tests:
  - a fold ending exactly on 2023-12-31 is accepted;
  - one ending on 2024-01-01 is refused;
  - a fold whose training part is valid but whose test window crosses the cutoff is refused;
  - an observation dated 2023-12-29 whose label ends in 2024 is refused.

### 6.2 Features come from an explicit allowlist

- `ml/allowlist.py` lists every approved feature column, table by table. Nothing is taken from a table
  automatically: a column not listed is refused, even in an approved table.
- Each entry records the source table, the column, the source version, the availability (`close_t`) and a
  feature version (the hash of the entry).
- Source versions:
  - `stock_features`: `FEATURE_SET_VERSION` (`v2`);
  - `stock_factors`: `factors.version()` (the hash of the factor definitions);
  - `market_regimes`: a new `regimes.version()`, the same kind of hash over the regime SQL, dimensions and
    default parameters. It is an additive function and changes no regime.
- A feature spec whose source version differs from the current one is refused (a definition changed since
  approval).
- Excluded sources are refused at validation with a named reason:
  - Interaction Engine outputs: scan statistics, not per-row values;
  - `report_marks` and fundamentals: no publication time;
  - Exposure and Risk outputs: portfolio-level;
  - `stock_targets`: outcomes, never features;
  - `cross.duckdb`: not approved in PR #10.
- The initial allowlist holds:
  - the 30 features of `stock_features` that the reference tests cover;
  - the 9 factors x `rank_pct` / `z` / `z_industry`;
  - the 6 regime states.
  The flags and identity columns (`is_traded`, `limit_up`, `session_index`, `period`, ...) are excluded.

### 6.3 Deterministic, versioned hashes

- Canonical serialization: JSON with sorted keys and fixed separators. Tuples become lists, dates become ISO
  strings, and floats use `repr`. A type outside this set is refused instead of being stringified.
- Every hashed payload carries `contract: "ml/1"`.
- The digest is SHA-256 (as `config_hash`), shown as 16 hex characters. Python's `hash()` is never used.
- Tests:
  - the same spec gives the same hash in two separate Python processes with different `PYTHONHASHSEED`
    values;
  - dict key order and date-versus-ISO-string inputs do not change the hash;
  - every meaningful field changes it (each one is tested);
  - the contract version changes it.

The acceptance in section 5 is extended accordingly. No other change to the plan.

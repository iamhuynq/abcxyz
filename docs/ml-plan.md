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

## 7. Leakage threats and the rule that blocks each (PR #10)

| Threat (spec) | Blocked by | Test |
|---|---|---|
| A feature known after its decision date | Only `close_t` features from approved point-in-time tables (allowlist, source version); `keys.check_point_in_time` refuses rows dated after their decision | `test_ml_contracts`: availability, source version, excluded sources, future feature rows |
| An outcome used as a feature | `stock_targets` and `feature_target` are excluded sources | excluded-source test |
| A source revised after the decision | Source versions: a feature approved under one definition is refused once it changes. Row-level revisions are covered by the integrity perturbation tests of the approved tables | source-version test |
| A label interval overlapping the test period | `purge` uses actual label end dates; `check_fold` raises if any training label reaches the embargo gap or the test window | `test_ml_splits`: purge with a suspended symbol, check_fold cases |
| Random shuffling | `SplitSpec.kind` is `expanding` or `rolling`; there is no shuffle field | split rules test |
| Validation or holdout data | `RESEARCH_END` invariant in `DatasetSpec`, `SplitSpec`, `folds` and `check_fold`, including label ends | cutoff tests (2023-12-31 accepted, 2024-01-01 refused, test window or label crossing refused) |
| Duplicate observations | `keys.check_unique` refuses them | key test |
| A non-reproducible spec | Canonical SHA-256 hashes with the contract version | hash tests across processes |
| Preprocessing fitted on the full dataset | Not in PR #10 (no preprocessing yet). PR #12 must fit on each fold's training rows only, with a test | to add in PR #12 |
| A universe member before its membership date, a stale exit price | The point-in-time universe and NULL incomplete labels already exist; the dataset builder must keep them | adversarial tests to add in PR #11 |

PR #11 adds the adversarial dataset tests:
- a feature row dated one session late;
- a revised source row;
- a stock before its first traded date;
- a missing exit price;
- a duplicate key.

## Result, PR #10 (2026-11-12)

Branch `ml-foundation`. Full suite: 616 passed (565 before). The only change outside `ml/` is an additive
`regimes.version()`; no engine behaviour and no stored data change.

- `ml/canonical.py`:
  - canonical JSON (sorted keys, fixed separators; ISO dates; tuples as lists; dataclasses by field; finite
    floats only; any other type refused);
  - contract `ml/1` wraps every payload; digest is SHA-256, 16 hex characters.
- `ml/allowlist.py`:
  - `stock_features`: the 30 reference-checked columns (a test ties the list to the reference test);
  - `stock_factors`: 27 entries (9 factors x `rank_pct` / `z` / `z_industry`);
  - `market_regimes`: the 6 states.
  - Source versions: `v2`, `f1:<factors.version()>` and `regimes.version()`.
  - Excluded sources are refused with a named reason; any other table or unlisted column is refused too.
- `ml/contracts.py`: `FeatureSpec`, `LabelSpec` (executable `stock_targets` labels only, with a matching
  horizon; entry `open_t+1`, exit `close_t+h`, gross, per-horizon masks), `DatasetSpec` (unique features, the
  point-in-time universe, fingerprint including the universe SQL), `SplitSpec` (`expanding` or `rolling`, no
  shuffle) and `ExperimentSpec` (hypothesis, approved model family, seed, Factor Engine baseline, declared
  primary test). The research cutoff is checked on every date.
- `ml/splits.py`:
  - `folds` (deterministic; complete test windows only);
  - `purge` (actual label ends);
  - `check_fold` (raises on any leak, including the cutoff and label ends);
  - `session_offset`.
- `ml/keys.py`: `check_unique`, `check_point_in_time`.
- Embargo semantics, made precise in the code: walk-forward training always precedes the test window, so the
  embargo is a gap of `embargo_sessions` market sessions **before** the test start. Section 4 said "after";
  that was an error, since nothing after the test window is ever trained on.
- Tests: `test_ml_contracts.py` (35) and `test_ml_splits.py` (16).
- Mutation check: 28 mutations, all caught.
  - One first survived: the split-kind check, because `random` was refused for another reason. The test now
    matches the message.
  - Four are stopped by a second defence in the code rather than the first assertion:
    - the fold guard after the cutoff filter;
    - the date-type check (comparing a datetime raises);
    - JSON `allow_nan=False` behind the finite-float check;
    - the minimum-training-rows check.

## PR #10 review (external, 2026-11-12, head `781d6be`): assessment and plan

All four findings were reproduced on the branch:
- mutating the caller's `params` dict changes `experiment_id`;
- mutating the `features` list passed in changes the dataset fingerprint (it is stored as a list);
- `hypothesis=None` raises `AttributeError`, not `ContractError`;
- a split ending in 2015 is accepted for a dataset that ends in 2023.

| Point | Assessment | Action |
|---|---|---|
| P1 specs are not deeply immutable | Valid | At construction, `params` becomes an immutable canonical form: nested dicts are converted to a read-only mapping (`MappingProxyType` of normalized values), lists to tuples, and unsupported types are refused. `DatasetSpec.features` is stored as a tuple of `FeatureSpec`, and anything else is refused. Tests: mutating the caller's dict or list (nested too) changes neither the spec nor its id, and the spec's own `params` cannot be modified. |
| P1 the Factor Engine baseline is not pinned | Valid | `BaselineSpec(factor, field = rank_pct, source_version)`, validated like a feature: the factor must exist and the version must be the current `f1:<factors.version()>`. `ExperimentSpec.baseline` is a `BaselineSpec`, so the id covers the baseline version. Test: a different baseline version changes the id, and a stale version is refused. |
| P2 dataset period and split boundaries are not cross-validated | Valid | `SplitSpec` gets a `first_date`. `ExperimentSpec` requires the split to cover the dataset exactly: `split.first_date == dataset.period_start` and `split.last_date == dataset.period_end` (no undeclared subset). `folds` uses only sessions in `[first_date, last_date]`. Tests: a split ending before the dataset end is refused, and a calendar starting before the dataset start yields no earlier session. |
| P2 purge and embargo boundaries | Valid as explicitness; no defect | Contract stated in the docstrings and section 7: training labels must end **strictly before** `embargo_start`, a market session `embargo_sessions` before the test start. Intervals come per symbol from the dataset builder, never from a market row count. `purge` also requires `label_start` and raises on an interval that does not start after its decision date, like `check_fold`. Boundary tests: a label ending the session before `embargo_start` is kept; one ending on it, inside the gap, or overlapping the test window is purged; an invalid interval raises. |
| 5.3 allowlist completeness | Valid | The test compares the full expected factor-column set (all 27 names), not only the count. |
| 5.4 type errors | Valid | Explicit type checks: every string field, integer field and date field raises `ContractError` on a wrong type. Test with `None` and numbers. |
| 5.1 / 5.2 FDR registration and the Newey-West test specification | Agreed; not in PR #10 | Recorded as requirements for PR #13 (section 8 below). |

## 8. Requirements carried to later PRs (from the PR #10 review)

- **FDR (PR #13).**
  - Exactly one primary test per experiment and per configuration, with a stable label.
  - The run starts, and the test is logged, before its result is read.
  - Failed or unfavourable runs stay in the run lifecycle (never deleted).
- **Primary-test specification, before any decision uses it.**
  - Rank IC per date (Spearman with tie handling, as the decile scan).
  - The incremental IC is the model IC minus the baseline IC on the same dates and rows.
  - Newey-West lag = horizon - 1, as `mean_test` elsewhere.
  - A minimum of 30 stocks per date and 60 valid dates per fold.
  - Dates with too few stocks are dropped and counted.
  - The baseline is pinned by `BaselineSpec`.
- **Dataset builder (PR #11).**
  - Per-symbol label start and end.
  - Allowlist and source versions only.
  - NULL labels excluded by an explicit, counted rule.
  - The fingerprint includes the data identity (warehouse SHA, feature build).

### PR #10 review fixes: result (2026-11-12)

Done as planned. Full suite: 624 passed (616 before).

- **Deep immutability.**
  - `ExperimentSpec.params` is frozen at construction: read-only mappings, tuples, and finite scalars or dates
    only.
  - `DatasetSpec.features` is copied into a tuple of `FeatureSpec`.
  - Tests: mutating the caller's dict or list (nested too) changes neither the spec nor its id; the spec's own
    params raise on assignment.
- **`BaselineSpec(factor, source_version, field)`.** The factor version is pinned to the current
  `f1:<factors.version()>`; a stale version is refused. A changed factor definition (simulated by patching
  `factors.version`) changes the experiment id.
- **`SplitSpec.first_date`.** The split must cover the dataset period exactly, and `folds` uses only sessions in
  `[first_date, last_date]`.
- **Purge contract.** Labels must end strictly before `embargo_start`. `purge` requires the label interval and
  raises on an invalid one.
  - Boundary tests: the session before the gap is kept; on `embargo_start`, inside the gap and in the test
    window are purged.
  - A row decided before a rolling training window is dropped.
- **Types.** Explicit type checks: `None`, numbers for strings, `True` or `1.0` for ints, and strings for specs
  raise `ContractError`.
- **Allowlist test.** It compares all 27 factor columns and all 6 regime states by name.
- **Mutation check.** 38 mutations (28 updated, 10 new), all caught. The three immutability mutations are caught
  by the immutability test on its own. Two first-pass findings got fixes:
  - the training-window purge had no row decided before a rolling window, so a test was added;
  - the "bool as int" mutation first disabled every type check instead of only the bool clause, so it was
    rewritten.

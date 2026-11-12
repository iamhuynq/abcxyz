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

## PR #10 second review (external, 2026-11-12, head `653b2c1`): assessment and plan

Both findings were reproduced:
- setting `regimes.TERCILE_DIMENSIONS` to another tuple leaves `regimes.version()` unchanged;
- `check_fold` accepts a fold that starts before the split and ends after `split.last_date` (2023-12-31 against
  2023-12-29), because both dates are before the research cutoff.

The audit of the other version inputs found the same gap in the factor source version:
- `factors.version()` hashes the factor SQL and definitions but not the `FactorParams` defaults (scoring
  universe ADV and sessions, minimum stocks, minimum industry size, winsor level), and all of them change the
  scores.
- `factors.version()` is also stored as the `pattern_version` of the recorded interaction run, so changing that
  function would relabel a stored run whose definitions did not change.

| Point | Action |
|---|---|
| `regimes.version()` omits `TERCILE_DIMENSIONS` | Add it to the hashed input. `regimes.version()` is new in this PR, so nothing stored uses it yet. The audit of `regimes.py` finds no other module-level input: `DIMENSIONS`, `TERCILE_DIMENSIONS`, `VALUES_SQL`, the three functions and the `RegimeParams` defaults are all hashed. Tests, each through `monkeypatch`: changing `TERCILE_DIMENSIONS`, `DIMENSIONS`, `VALUES_SQL` or a `RegimeParams` default changes the version. |
| Factor source version omits `FactorParams` | The ML source version of `stock_factors` becomes `f1:<factors.version()>:<digest of FactorParams()>`. `factors.version()` itself is unchanged, so the stored interaction run keeps its meaning. Test: changing a `FactorParams` default (for example `winsor`) changes the ML source version, so the feature and the baseline are refused until approved again. |
| `check_fold` does not validate the fold against the split | `check_fold` checks `fold.train_start >= split.first_date` and `fold.test_end <= split.last_date`, and the ordering `train_start < embargo_start <= test_start`, `train_end < test_start <= test_end`. Tests: a fold starting before the split, one ending after `split.last_date` but before the cutoff, and each ordering violation are refused. |
| CI | Confirmed by the user from the run of the pushed head. |

### PR #10 second review fixes: result (2026-11-12)

Done as planned, with the reviewer's additional conditions. Full suite: 638 passed (624 before).

- **`regimes.version()`.** It now hashes canonical JSON (sorted keys, fixed separators) of:
  - the values SQL and the source of the percentile, state and build functions (the risk rule lives in
    `build_regimes`);
  - `DIMENSIONS` (states and descriptions) and `TERCILE_DIMENSIONS`;
  - every `RegimeParams` default, via `asdict`.

  `repr()` is no longer used. The audit found no other module-level input in `regimes.py`.
  - Tests: changing `TERCILE_DIMENSIONS`, `DIMENSIONS`, `VALUES_SQL` or a `RegimeParams` default (two of them)
    changes the version, each tested independently; reordering `DIMENSIONS` does not.
- **`allowlist.factor_source_version()`.** It is the single source used by both `FeatureSpec` and
  `BaselineSpec`: `f1:<factors.version()>:<canonical digest of every FactorParams field>`.
  - `factors.version()` is unchanged, so the recorded interaction run keeps its meaning.
  - Test, parametrized over every `FactorParams` field: changing that default changes the ML version, keeps
    the `f1:<factors.version()>` prefix, and refuses a feature or a baseline built with the old version.
- **`check_fold` is an independent guard.** It checks the fold itself, not only the rows: all five boundaries
  must lie in `[split.first_date, split.last_date]`, with `train_start <= train_end < test_start <= test_end`
  and `train_start < embargo_start <= test_start`.
  - Tests cover a fold correctly ordered but starting before the split, one ending after `split.last_date` yet
    before the cutoff, and each ordering and embargo violation.
  - Every fold that `folds()` produces (expanding, rolling, embargo 0) passes the same check.
- **Mutation check.** 47 mutations (38 earlier plus 9 new), all caught by an assertion.

## PR #10 third review (external, 2026-11-12, head `116dacc`): assessment and plan

**Verified.**
- `build(..., factor_params, regime_params)` accepts overrides and forwards them to `build_factors` and
  `build_regimes`, but `feature_builds` records neither.
- `quant build` (`cli.py`) always uses the defaults; only tests pass other values. So no production build is
  known to be non-default, but the database cannot prove it.
- The ML source versions hash the code defaults, not the parameters that built the database. A build made with
  other parameters would be accepted under the default version.

**Decision: option A (persist the effective configuration), with option B's strictness for unknown builds.**

1. **`build.py`.** `feature_builds` gets four nullable columns:
   - `factor_params`: canonical JSON of the `FactorParams` actually used;
   - `regime_params`: the same for `RegimeParams`;
   - `factors_version`: `factors.version()` at build time;
   - `regimes_version`: the regime definition version at build time.

   The insert names its columns. A fresh build writes them. Older history rows are carried over `BY NAME`, so
   they keep NULL, which means unknown.
2. **Regime definition version.** `regimes.version()` stops hashing the `RegimeParams` defaults: it becomes the
   definitions only (SQL, code, `DIMENSIONS`, `TERCILE_DIMENSIONS`), like `factors.version()`. The parameters
   enter the ML source version separately, as for factors. `regimes.version()` is new in this PR, so nothing
   stored depends on it.
3. **ML source versions as a function of definitions plus parameters.**
   - `allowlist.factor_source_version(params)` = `f1:<factors.version()>:<digest(params)>`.
   - `allowlist.regime_source_version(params)` = `<regimes.version()>:<digest(params)>`.
   - Approved means the current definitions with the default parameters. `FeatureSpec` and `BaselineSpec`
     accept only the approved versions, as now.
4. **New `ml/builds.py`: `build_source_versions(con, build_id)`.** It reads the stored row and returns the
   effective version of each approved table:
   - `stock_features` gets `feature_set_version`;
   - `stock_factors` and `market_regimes` get their version from the stored definitions and parameters.

   It raises when the row is missing or any of the four columns is NULL (a legacy or unknown build is never
   treated as default). `check_build(con, build_id, dataset)` refuses a dataset when any feature's approved
   version differs from the build's effective version. PR #11 calls it before reading any feature.
5. **`factors.version()` is unchanged.** A test pins that it does not depend on `FactorParams`.

**Tests**, on synthetic builds:
- the default build records the approved versions, and `check_build` accepts it;
- a build with a changed `FactorParams` value records different metadata, and `check_build` refuses the
  factor features and the baseline factor;
- the same with a changed `RegimeParams` value, refused for the regime features;
- a legacy row (the four columns NULL) is refused;
- a rebuild keeps the history rows;
- `factors.version()` is unchanged by `FactorParams`.

**Consequence for the real store.** The current `research.duckdb` was built before these columns existed, so ML
refuses it until the next `quant build`, which runs with the defaults and records them. The daily pipeline
rebuilds it.

### PR #10 third review fixes: result (2026-11-12)

Done as planned. Full suite: 646 passed (638 before).

- **`feature_builds` records the effective configuration.** Four nullable columns: `factor_params` and
  `regime_params` (canonical JSON of the parameters actually passed to `build()`), `factors_version` and
  `regimes_version`. The insert names its columns. Older rows are carried over `BY NAME` with NULL.
- **Definitions are split from parameters.**
  - `regimes.version()` now hashes the definitions only (SQL, code, `DIMENSIONS`, `TERCILE_DIMENSIONS`), like
    `factors.version()`, which is unchanged.
  - The ML source versions are `factor_source_version(params, definitions)` and
    `regime_source_version(params, definitions)`. Their defaults (current definitions, default parameters) are
    the approved versions; `FeatureSpec` and `BaselineSpec` use them.
- **`ml/builds.py`.**
  - `build_source_versions(con, build_id, schema)` computes each table's effective version from the stored row.
  - It raises `ContractError` for a missing build, a row with NULL metadata, or a `feature_builds` table without
    the columns at all.
  - `check_build(con, build_id, dataset, baseline)` refuses any feature, or the baseline, whose approved version
    differs from the build's.
- **Found while checking the real store.** The real `research.duckdb` has the old schema without the new
  columns. The first version of `build_source_versions` raised a DuckDB `BinderException` there instead of a
  `ContractError`, so the columns are now checked first, with a test on an old-schema table.
  - The real build is refused with "feature_builds has no factor_params, ... rebuild with `quant build`", as
    planned. The next `quant build` records the defaults.
- **Tests (`test_ml_builds.py`, 8).**
  - A default build is recorded and accepted.
  - A build with other `FactorParams` is refused for factor features and for the baseline, while its default
    regimes are accepted; a build with other `RegimeParams` is refused for regime features.
  - NULL metadata, the old schema and a missing build are all refused.
  - A rebuild keeps history rows (legacy NULL).
  - The schema-attached database works.
  - `factors.version()` does not depend on `FactorParams`.
- **Mutation check.** 55 mutations, all caught: 47 earlier, 2 re-anchored, the regime-params mutation retired
  (parameters left that hash by design), and 9 new.
  - The old-schema mutation is caught by DuckDB's error, which is exactly the failure the check prevents.

## PR #10 fourth review point (external, 2026-11-12): factor implementation not in the source version

**Verified, and wider than reported.**
- `factors.version()` hashes `RAW_SQL`, `NORMALIZE_SQL` and `FACTORS`, but not `build_factors()`, which formats
  the parameters into the SQL and chooses the factor columns.
- The same class of gap exists elsewhere:
  - the `stock_features` source version is the constant `v2`, unchanged by an edit of
    `03_stock_features.sql`;
  - factors and regimes are computed from `daily_panel` and `stock_features`, so an upstream SQL change alters
    them while no ML version changes.
- `ml/builds.py` ignores `feature_builds.code_hash`, which already identifies all of this:
  `feature_build_hash()` hashes every build SQL file and the whole of `build.py`, `factors.py` (so
  `build_factors`), `regimes.py` and `backtest/costs.py`. It is non-NULL on every build row.

**Plan.** Bind every ML source version to the feature-build implementation.
- `source version = <table definition>:<parameter digest>@<feature build code hash>`:
  - `stock_features`: `v2@<code>`;
  - `stock_factors`: `f1:<factors.version()>:<FactorParams digest>@<code>`;
  - `market_regimes`: `<regimes.version()>:<RegimeParams digest>@<code>`.
- The approved version uses the current `feature_build_hash()`. A build's effective version uses its stored
  `code_hash`.
- So any change to the code that produces `research.duckdb` changes the approved versions (and every experiment
  id). A build made by other code is refused by `check_build` until rebuilt.
- No new column; `factors.version()` is unchanged.
- This is deliberately coarse: an edit anywhere in the build, comments included, requires a rebuild before ML
  accepts the data. That is the safe direction for provenance.

**Tests.**
- A changed build code hash (patched `feature_build_hash`) changes all three approved versions, and refuses a
  feature or baseline built under the old one.
- A build row whose `code_hash` differs from the current code is refused.
- `factors.py`, the module that defines `build_factors`, and every build SQL file are in
  `feature_build_files()`.
- A default build made by the current code is accepted.

### PR #10 fourth review point: result (2026-11-12)

Done as planned. Full suite: 649 passed (646 before).

- **Every ML source version ends with `@<feature build code hash>`.**
  - Versions: `stock_features` `v2@<code>`; `stock_factors` `f1:<factors.version()>:<FactorParams digest>@<code>`;
    `market_regimes` `<regimes.version()>:<RegimeParams digest>@<code>`.
  - Approved versions use the current `feature_build_hash()`; `ml/builds.py` uses the build's stored `code_hash`.
  - So `build_factors`, every build SQL file (upstream `daily_panel` and `stock_features` included),
    `regimes.py` and `costs.py` are all bound. `factors.version()` is unchanged, and no column was added.
- **Tests.**
  - A changed build code hash changes all three approved versions and leaves the definition and parameter parts
    as they were.
  - Features and the baseline approved under the old code are refused.
  - A build row with another `code_hash` is refused.
  - `factors.py` (where `build_factors` lives) and every build SQL file are in `feature_build_files()`.
- **Mutation check.** 60 mutations (55 earlier, four re-anchored, and 5 new), all caught.

### PR #10 fifth review point: result (2026-11-12)

An empty build provenance value is refused, never replaced.
- `build_source_versions` refuses NULL, non-string, empty and blank values in all six metadata fields
  (`factor_params`, `regime_params`, `factors_version`, `regimes_version`, `feature_set_version`, `code_hash`).
- The version helpers substitute the current value only when the argument is omitted (`None`); an explicit
  empty string raises.
- Tests:
  - 12 parametrized cases: each field set to `''` and `'   '`;
  - explicit empty arguments to the helpers.
- Both mutations are caught: checking only `None` in builds, and dropping the empty-value check in the helpers.
  The full suite has 662 passing tests.

## PR #11: dataset builder (plan, 2026-11-12)

Source: the external spec "PR #11 - ML Dataset Builder" (branch `ml-dataset-builder` from `3bbbda5`).

### Audit

- `pyarrow` is already a dependency, so Parquet needs nothing new.
- `stock_targets` does not expose the entry and exit dates: `d1` to `d20` are computed inside
  `04_stock_targets.sql` and dropped.
- The masks exist per horizon: `fwd_has_price_jump_{1,3,5,10,20}d` and `crosses_period_{1,3,5,10,20}d`, plus
  `entry_blocked`.
- Feature rows are dated by their decision date. There is no separate availability column: point in time comes
  from the tested SQL, the allowlist and the source versions.
- The real `research.duckdb` still has the old `feature_builds` schema, so ML refuses it until the next
  `quant build`.

### Decisions (to approve)

1. **Label intervals without approximation and without changing the target SQL.**
   - The builder derives `label_start` (`lead(date, 1)`) and `label_end` (`lead(date, h)`) from `daily_panel`
     with the same window as `04_stock_targets.sql`: `PARTITION BY symbol ORDER BY date`, the symbol's own rows.
   - A test checks, on the synthetic build, that the interval reproduces the label: `close[label_end] /
     open[label_start] - 1 == fwd_ret_exec_{h}d`. It also checks a symbol with gaps (the dropped zero-price row)
     and the last sessions.
   - `stock_targets` itself is not changed; changing it would change the feature-build code hash and every
     source version.
2. **Exclusions are sequential**, each row under its first reason, in this order, with input = retained + sum:
   1. missing label interval (no `h`-th session);
   2. NULL executable label;
   3. `entry_blocked`;
   4. `fwd_has_price_jump_{h}d`;
   5. `crosses_period_{h}d`, or a label end past the dataset period or the research cutoff;
   6. outside the point-in-time universe (`ResearchParams().point_in_time_sql()`, the `DatasetSpec` universe);
   7. a NULL required feature.

   Overlapping counts are reported as diagnostics. No row is dropped any other way, and nothing is filled.
   NULL features are excluded by rule 7, not imputed.
3. **API.** `build_dataset(con, spec, build_id, schema="rs.") -> DatasetResult(data, metadata, fingerprint,
   exclusions)`.
   - `check_build` runs first.
   - Identifiers come only from the validated specs; values are bound as parameters.
   - `check_unique` and `check_point_in_time` (`feature_date` = the source row's date) run on the result.
4. **Artifact (optional).** `write_dataset(result, path)` writes `<path>.parquet` and `<path>.json` (metadata)
   atomically (temporary file, then rename). It refuses an existing path. The fingerprint covers the canonical
   logical content (sorted rows, every value) and the provenance, not the Parquet bytes.
5. **Fingerprint and metadata.**
   - The spec fingerprint, ordered features with their versions, label and horizon, `build_id`, the effective
     source versions, `code_hash`, `warehouse_sha`, `data_as_of`, row counts, exclusions and `contract: ml/1`.
   - No wall-clock time.
   - Empty or NULL provenance fails closed (already in `ml/builds.py`; `warehouse_sha` is added to that check).
6. **Read-only.** Every database is opened read-only. No table, view or row is written.

### Tests

Tests use synthetic builds and cover the spec's list: provenance, row integrity, labels and intervals,
exclusion accounting, and artifact behaviour. Every new test is mutation-checked.

### PR #11 plan review (external, 2026-11-12): revisions

**Checked in `04_stock_targets.sql` (the reviewer's blocker).**
- `fwd_ret_exec_{h}d = lead(adj_close, h) / lead(adj_open, 1) - 1`, over `daily_panel` with
  `PARTITION BY symbol ORDER BY date` (lines 10-12, 26, 37). The builder reads the same table and columns, so the
  prices come from the same source.
- `entry_blocked`, `fwd_has_price_jump_{h}d` and `crosses_period_{h}d` are separate flags (lines 48, 59, 71).
  None of them makes the label NULL.
- A label is NULL only when the symbol has no h-th later row. `daily_panel` drops non-positive prices, so there
  is no other way. The masks are therefore never hidden by a NULL label. The revised order puts the known causes
  first anyway, and a test proves that no masked row has a NULL label.

**1. Interval definition (as the target SQL, which the label must reproduce).**
- `label_start = lead(date, 1)`: the symbol's own next row after t (entry at its open).
- `label_end = lead(date, h)`: the symbol's h-th own row after t (exit at its close). That is the (h-1)-th row
  after `label_start`, not the h-th: `fwd_ret_exec_3d` enters at the open of t+1 and exits at the close of t+3.
  The reviewer's wording ("h-th from `label_start`") would not reproduce the stored label.
- The order comes from each symbol's own `daily_panel` rows, never a market-wide index.
- If the h-th row is missing, there is no interval. That is reason 2, never inferred.
- Test: `adj_close(label_end) / adj_open(label_start) - 1 == fwd_ret_exec_{h}d` to 1e-12, for every symbol and
  every horizon (3, 5, 10, 20) of the synthetic build. That covers a late first trade (DEL, UPC), a row dropped
  inside the series (STK's zero-price placeholder) and the series ends.

**2. Exclusions: one status per row, first match wins, the order explicit in code.** A row is either `kept` or
has exactly one `exclusion_reason`, assigned by an ordered tuple of named predicates evaluated in that order:
1. `outside_universe`: fails the point-in-time universe at t (eligibility uses only data at t, so it comes first);
2. `no_label_interval`: the symbol has no h-th later row (this is also when the label is NULL);
3. `entry_blocked`;
4. `price_jump_in_window`: `fwd_has_price_jump_{h}d`;
5. `label_crosses_period`: `crosses_period_{h}d` (for a research dataset, this includes every label that starts
   before 2023-12-31 and ends after it);
6. `label_after_dataset_end`: `label_end` after `DatasetSpec.period_end` (within research, when the dataset
   ends earlier);
7. `label_null`: residual, a NULL label with an interval (expected 0; counted, never filled);
8. `feature_null`: a required feature is NULL.

The table gives the count and the share of input rows for each reason, plus `input_rows = kept_rows +
excluded_rows`. A test proves the reconciliation and that every row has exactly one status.

**3. Universe: point in time, from row data at t only.**
- It is `ResearchParams().point_in_time_sql()`: traded at t, ADV20 at t > 1 bn, more than 20 own sessions, no
  price jump at t, not the bad source date.
- `daily_panel` keeps delisted stocks (459 covered, `audit pit`), so no current membership list is used.
- Stated limitation (integrity finding E-1): exchange and industry are current values. The universe filter does
  not use them, but industry-based features (`z_industry`) do.
- The universe SQL text is in the `DatasetSpec` fingerprint (PR #10) and in the metadata.

**4. API.** `build_dataset(con, spec, build_id, schema) -> DatasetResult`, with:
- `data` (kept rows: keys, features in spec order, label, `label_start`, `label_end`);
- `metadata` (read-only mapping);
- `fingerprint`;
- `exclusions` (ordered: reason, count, share);
- `input_rows`, `kept_rows`, `excluded_rows`.

`check_build` runs before any dataset read and checks every source the spec uses; `warehouse_sha` is added to
the non-empty provenance check. It never runs `quant build`. A test compares the database file's hash before and
after the call.

**5. Artifact: one atomic directory.**
- `write_dataset(result, path)` writes `data.parquet` and `metadata.json` into a temporary sibling directory,
  verifies both, and renames the directory to `path` in one step. `path` must not exist.
- A failure before the rename leaves no `path`, and the temporary directory is removed.
- `read_dataset(path)` recomputes the content hash and refuses a mismatch with the metadata fingerprint.
- Tests:
  - the target already exists;
  - a failure while writing the metadata;
  - a failure before the rename (patched);
  - a tampered value or a mismatched fingerprint is refused on read.

**6. Fingerprint = canonical hash of provenance plus logical content.**
- Provenance: contract `ml/1`, dataset schema version `dataset/1`, spec fingerprint, ordered columns with
  feature versions, label and horizon, `build_id`, effective source versions, `code_hash`, `warehouse_sha`,
  `data_as_of`, the ordered exclusion counts, and the row count.
- Content: rows sorted by `(date, symbol)`, each serialized canonically (ISO dates, floats as `float.hex`, NULL
  as null, string labels as-is) and fed into SHA-256 in column order.
- Tests:
  - changing one feature value, the column order, the label, the build or the warehouse SHA changes the
    fingerprint;
  - identical inputs give the same fingerprint, across processes as well.

### PR #11: final clarifications before implementation (2026-11-12, plan approved)

1. **`outside_universe` is reachable and is not a join condition.**
   - Candidate rows are every `daily_panel` row `(date, symbol)` in the dataset period: every session from the
     symbol's first to its last traded date, including no-trade sessions.
   - The universe is `ResearchParams().point_in_time_sql()`, evaluated on that row's `stock_features` values at
     t: traded at t, `adv_value_20 > 1 bn`, `session_index > 20`, no price jump at t, not 2025-07-16.
   - A candidate fails it on a no-trade day, when illiquid, in its first 20 sessions, or on a price-jump day.
     These cases are frequent, so the reason is real.
   - Features are left-joined on `(date, symbol)`. A missing feature value falls under `feature_null`, never a
     silent drop.
   - Source and version of the universe: the SQL text, already in the `DatasetSpec` fingerprint, and the build.
2. **`z_industry` is removed from the default allowlist.** The 9 `<factor>.z_industry` entries are removed until
   a point-in-time industry history exists, because they use the current ICB classification. The limitation is
   stated in `allowlist.py`, in the dataset metadata (`limitations`) and here. The allowlist test is updated
   (18 factor entries).
3. **Every exclusion reason is always in the metadata, in the declared order, count 0 included.** Test: a
   dataset where some reasons do not occur still lists all 8, and the counts reconcile with the input rows.
4. **Intervals come from the full series.** `label_start` and `label_end` are computed with `lead()` over each
   symbol's complete `daily_panel` series (all its rows, before any period filter or exclusion), and only then
   joined to the candidates.
   - A test removes candidate rows (a dataset period that starts mid-series, and excluded rows) and checks that
     the intervals of the kept rows are unchanged and still reproduce `fwd_ret_exec_{h}d`.

Scope unchanged: no model training, no experiment runner.

### Result, PR #11 (2026-11-12)

Branch `ml-dataset-builder`. Full suite: 677 passed (662 before). No database is written, no model,
no runner.

- **`ml/dataset.py`.**
  - `build_dataset(con, spec, build_id, schema)`:
    - runs `check_build` first, then `build_identity` (non-empty `warehouse_sha` and `code_hash`, plus
      `data_as_of`);
    - candidates are every `daily_panel` row in the period;
    - intervals are `lead(date, 1)` / `lead(date, h)` over each symbol's full series;
    - features are left-joined from the validated specs only (factor names bound as parameters);
    - `check_unique` and `check_point_in_time` run on the result;
    - every row gets one status in the explicit `REASONS` order;
    - accounting is reconciled or the call raises.
  - `DatasetResult`: data, metadata, fingerprint, exclusions (every reason, zeros included, with its share),
    and the row counts.
  - Fingerprint: the canonical digest of the provenance plus `content_hash` (rows sorted by date and symbol,
    floats as `float.hex`).
  - `write_dataset` / `read_dataset`: one temporary directory holding `data.parquet` and `metadata.json`,
    verified and renamed in one step. An existing path is refused, and a failure leaves nothing behind. Reading
    refuses a content-hash or fingerprint mismatch.
- **`z_industry` removed from the allowlist** (18 factor entries). `allowlist.LIMITATIONS` is copied into the
  metadata.
- **`ml/builds.build_identity`** checks the data identity of a build.
- **Tests (`test_ml_dataset.py`, 15).**
  - The read-only check compares the database file hash before and after.
  - Interval and label reproduction to 1e-12, for every symbol and horizons 3/5/10/20, against each symbol's
    own sessions. This includes a no-trade session inside UPC's series, and a dataset starting mid-series
    (intervals unchanged).
  - Masks never hidden by a NULL label; first match wins; `feature_null` counted; `label_after_dataset_end`.
  - Provenance: empty `warehouse_sha`, non-approved parameters, missing build.
  - `z_industry` and target columns refused.
  - Fingerprint: content, order, label, warehouse SHA; stable across processes.
  - Artifact: atomic, never overwritten, a failure while writing the metadata or before the rename leaves
    nothing, and tampered data or metadata is refused.
- **Mutation check.** 13 mutations, all caught. Three first survived and got tests:
  - intervals computed from traded rows only: a no-trade session was added to the fixture;
  - the fingerprint not checked on read: a tampered-metadata test;
  - the dataset-end rule: a dataset ending mid-series.

### PR #11 review fixes: result (2026-11-12)

Full suite: 687 passed (677 before). Scope unchanged: no model, no runner.

- **P1-1: fingerprint bound to content.**
  - `dataset.fingerprint(provenance, content)` is a canonical SHA-256 over the metadata (without
    `content_hash`) and the logical content hash. The same function is used by `build_dataset` and
    `read_dataset`.
  - On read, the content hash is recomputed from the data, compared with the metadata, and then used to
    recompute the fingerprint.
  - This is an **integrity check, not a digital signature**: it detects any change to the data, the content
    hash, the metadata or the fingerprint, but whoever can rewrite all of them can recompute it. Authenticity
    would need a key, which is out of scope.
  - Tests:
    - data and `content_hash` rewritten together with a stale fingerprint are refused, which shows the
      content is in the fingerprint;
    - a tampered `content_hash`, a changed metadata field and a forged fingerprint are each refused.
- **P1-2: period crossing from the interval itself.**
  - The builder takes `decision_period` and `lead(period, h)` (the period of `label_end`) from
    `daily_panel`'s own period column, over the same full series.
  - `label_crosses_period` = the two periods differ, or `label_end` > 2023-12-31. `stock_targets`'s
    `crosses_period` is no longer read.
  - `label_after_dataset_end` stays separate.
  - A kept label ending after the cutoff raises (an invariant).
  - Tests:
    - a build running from 2023-06 into 2024: every in-universe row whose label ends in 2024 is
      `label_crosses_period`, a label ending on 2023-12-29 is not, and no kept label ends after the cutoff;
    - a hand-made frame checks each rule (period change, cutoff) on its own.
- **Point in time.** Missing feature dates are no longer filled with the decision date. Dated rows are checked
  (a late feature raises); rows without a `stock_features` row are counted in
  `metadata.rows_without_stock_features`.
- **`_cell`.**
  - NumPy integers become `int`; floats become `float.hex`, with -0.0 as 0.0 and +/-inf as `inf` / `-inf`.
  - NaN, None and NaT are null; NumPy and Python booleans become `bool`.
  - A midnight Timestamp, a `date` or a `datetime64` gives an ISO date; a datetime with a time gives full ISO.
  - Any other type raises.
- **Concurrent publishers.**
  - `write_dataset` takes an `O_EXCL` lock file per path, which is atomic across processes and threads.
  - It refuses while another publisher holds the lock and refuses an existing destination; the lock and the
    temporary directory are removed in every case.
  - Test: 4 threads publish to one path; exactly 1 succeeds, the 3 others are refused, and the artifact verifies.
- **Excluded candidates.** Test: for every excluded row with an interval (outside the universe included),
  `label_start` and `label_end` equal the symbol's own next and h-th `daily_panel` sessions.
- **Mutation check.** 20 mutations, all caught (13 earlier, re-anchored, and 7 new).
  - One first survived: the period comparison, because for a research-only dataset every crossing is also past
    the cutoff and the two rules masked each other. The hand-made test now checks each rule on its own, and
    removing either rule is caught.

### PR #11 second review: result (2026-11-12)

Full suite: 688 passed (687 before).

- **P2-1: stale publish lock.**
  - The lock file records `{pid, host, created, token}`.
  - A lock is recovered only when it is on the same host and its pid no longer runs. Recovery renames the lock to
    a unique name, re-reads it, and gives it back if its content changed (a live lock taken in between), so a
    live lock is never removed. Age is never used.
  - A lock that cannot be checked (another host, unreadable content) is kept, and the error gives the manual
    recovery: make sure no publisher runs, then delete the `.lock` file.
  - Test: a lock left by an exited process is recovered and the artifact is published; a lock held by a live pid,
    or written on another host, is kept unchanged and refused.
- **P2-2: `np.datetime64("NaT")`** canonicalizes to null, with a regression assertion.
- **Mutation check.** 24 mutations, all caught (20 earlier plus 4 new): NaT not null, stale lock never
  recovered, live lock treated as stale, other host trusted.
- CI is confirmed by the user on the pushed head.

### PR #11 third review: result (2026-11-12)

Full suite: 689 passed (688 before).

- **P2: race in stale-lock recovery (confirmed).**
  - `os.rename(lock, moved)` does not guarantee that the file renamed is the one just read. If publisher B reads a
    stale lock, publisher A recovers it first and takes a new live lock, B then renames A's live lock, fails to
    restore it because C has taken the path, and deletes it while A is still publishing.
  - This cannot be closed with PID files and renames, so the recovery code is removed.
- **Fix: an OS advisory lock.**
  - `write_dataset` opens `.{name}.lock` (created if missing) and takes `fcntl.flock(LOCK_EX | LOCK_NB)`.
  - It holds the lock for the whole publish and closes the descriptor in `finally`, which releases the lock.
  - The kernel releases the lock when its holder exits, crashes or is killed, so there is no stale lock, no PID
    or host bookkeeping, and no manual recovery.
  - The lock file is never deleted. Deleting a flock file while another process holds it open would let two
    publishers lock different inodes. The leftover empty `.lock` file is harmless.
  - The file records `{pid, host, since}` for diagnostics only; nothing reads it.
  - `flock` locks per open file description, so threads of one process also exclude each other (macOS and Linux).
    POSIX only; the project runs on macOS.
- **Tests.**
  - 4 concurrent threads: exactly 1 publishes and 3 are refused. The lock file stays and no temporary directory
    is left. A leftover lock file that nobody holds does not block a publish.
  - A subprocess holds the lock:
    - publishing is refused 3 times;
    - the lock file keeps the same inode and the holder still holds it (a non-blocking flock from the test fails),
      so a live lock is never removed;
    - after SIGKILL (no cleanup), the kernel frees the lock and publishing succeeds; the lock file inode is
      unchanged.
  - A publish that raises still releases the lock: the next publish succeeds.
- **Mutation check.** 25 mutations, all caught:
  - 20 earlier dataset mutations; the 4 PID-recovery mutations no longer apply and are retired;
  - 5 new: no flock, shared instead of exclusive lock, lock file unlinked after publish, descriptor never
    closed, lock file replaced on acquire.

## PR #12: baseline models (plan, 2026-11-12)

Input: the external PR #12 plan (models, training, prediction, artifacts). Branch `ml-baseline-models` from
`main` at `f06945b` (PR #11 merged). This section maps that plan onto the existing contracts and records where
it differs from earlier decisions.

### Audit

- PR #11 is merged (`f06945b`); `main` equals `origin/main`; 689 tests pass.
- `DatasetResult.data` holds the kept rows only: `date`, `symbol`, the features in `DatasetSpec` order, the
  label, `label_start`, `label_end`. Kept rows have no NULL feature or label (excluded as `feature_null` /
  `label_null`). Rows are written in (`date`, `symbol`) order; `content_hash` sorts by that key, so row
  positions are well defined only if that order is checked.
- `contracts.MODELS = ("ridge", "elasticnet")`; `ExperimentSpec(model, params, seed, ...)` already validates
  model families, a seed and deep-frozen params.
- Dependencies: numpy, pandas, scipy, statsmodels. No scikit-learn and no tree library. Decision 7 of this plan
  (section 3) requires a written dependency review before any tree library.
- `provenance.ENV_PACKAGES` and `environment()` already record library versions; `canonical.digest` is the
  hashing convention; `dataset.py` holds the atomic publish and the `flock` lock.

### Differences from the external plan (to approve)

1. **Models without new dependencies.** The external plan names `DummyRegressor`, `Ridge` and
   `HistGradientBoostingRegressor` (scikit-learn). Recommendation:
   - `mean`: predicts the training label mean. A sanity reference, not an experiment model: a separate
     `REFERENCE_MODELS` tuple, so `ExperimentSpec` still accepts only `ridge` and `elasticnet` and its
     `experiment_id` contract is unchanged.
   - `ridge`: closed form in numpy (`solve((X'X + alpha I), X'y)` on standardized features, intercept not
     penalized).
   - `elasticnet`: statsmodels `OLS.fit_regularized(method="elastic_net", alpha, L1_wt)` on standardized
     features, deterministic coordinate descent.
   - Tree model: deferred. It needs the dependency review (size, license, determinism across threads) and a
     pickle artifact (see 2); it can be its own small PR after PR #13 if a linear model shows anything.
2. **Artifact without pickle.** Every model above is fully described by numbers: training means and
   standard deviations, coefficients, intercept. The artifact stores them as canonical JSON (floats as
   `float.hex`), so loading executes no code and the serialization trust boundary of the external plan does not
   arise. The fitted model is rebuilt from those numbers and must predict identically.
3. **Model spec.** A new frozen `ModelSpec(model, params, seed)` in `contracts.py`, validated like the other
   specs: known model, allowed parameter names per model (`ridge`: `alpha > 0`; `elasticnet`: `alpha > 0`,
   `0 < l1_wt <= 1`; `mean`: none), seed >= 0. Target and features are not repeated in the spec: they come from
   the dataset's own metadata (`DatasetSpec`), so they cannot disagree. Prediction then checks the dataset's
   ordered features (names and source versions) and label against the trained model.
4. **Dataset policy for prediction.** A trained model predicts only on the dataset it was trained on (same
   fingerprint). This is all PR #13 needs (each fold trains and predicts on one dataset). Prediction on another
   dataset (forward data) needs its own rule and a pre-registration later; it is refused now.
5. **Shared publish code.** The atomic directory publish and the `flock` lock move from `dataset.py` to
   `ml/publish.py`, used by both the dataset and the model artifact. No behaviour change: the existing dataset
   tests, including the lock tests, must pass unchanged.

### Design

- `contracts.py`: `ModelSpec`, `REFERENCE_MODELS`.
- `models.py`: the three estimators as pure functions over numpy arrays: `fit(spec, X, y) -> FittedModel` (a
  frozen record of means, scales, coefficients, intercept) and `predict(fitted, X)`. Standardization uses the
  training rows only (ddof 0). A feature constant in the training rows gets scale 1 and, centered, contributes
  nothing (ridge gives it coefficient 0); this is recorded, not refused.
- `training.py`:
  - `fit_model(dataset, train_indices, spec) -> TrainedModel`;
  - `predict_model(trained, dataset, prediction_indices) -> DataFrame`.
  - Indices are row positions in `dataset.data`. Checks, all failing with `ContractError`:
    - the dataset fingerprint verifies (as `read_dataset` does) and rows are in (`date`, `symbol`) order with
      unique keys;
    - indices are a non-empty 1-D integer sequence, no bool, no duplicates, no negatives, within range;
    - the feature matrix is exactly the dataset's ordered feature columns: no label, interval, key or metadata
      column can enter it, by construction and by a check;
    - every feature and label value used is finite (no NaN or inf), never filled;
    - for prediction: same dataset fingerprint, same ordered features and label as the trained model.
  - The feature matrix is built from the selected rows only, so nothing outside `train_indices` can reach the
    fit. Prediction reads only feature columns.
  - Training metadata: row count, a hash of the ordered training keys, first and last training dates and the
    latest training `label_end` (all from the rows, no calendar arithmetic), so PR #13 can check its folds.
  - Output of prediction: `date`, `symbol`, `label`, `prediction`, `model_id`, `dataset_fingerprint`, in the
    order of `prediction_indices`.
- `artifacts.py`: `save_model(trained, path)` and `load_model(path)`: one directory with `model.json`
  (`{"model_id", "metadata", "fitted"}`), published atomically, never overwritten. Metadata: artifact schema
  version, `ModelSpec` and its version, label and horizon, ordered features with source versions, dataset
  fingerprint and its provenance (copied from the dataset metadata), training summary, library versions
  (`provenance.environment()`), a hash of the ML code files, `created_at`. `model_id` is a canonical digest of
  everything except `created_at`. `load_model` recomputes it and refuses any edit, an unknown schema version, or
  a fitted record whose shapes do not match the features.
- No change to the Factor Engine, the Risk Engine, the warehouse schema, the dataset builder's behaviour or any
  portfolio code. No folds, no purge, no performance metrics (PR #13 and #14).

### Tests (`tests/research/test_ml_models.py`, `test_ml_training.py`)

- Estimators: ridge against an independent least-squares solution on augmented rows; elasticnet with
  `l1_wt = 1` zeroes a pure-noise feature on a hand-made problem; `mean` equals the training mean; a constant
  feature gets coefficient 0.
- `ModelSpec`: unknown model, unknown or invalid parameter, negative seed, bool where an int is expected.
- Indices: empty, duplicate, negative, out of range, float, bool, 2-D are refused.
- Leakage and adversarial:
  - changing labels of prediction rows does not change predictions;
  - changing features or labels of rows outside `train_indices` does not change the fitted record;
  - standardization statistics equal those of the training rows alone;
  - a reordered or renamed feature column, a dataset with a label column added as a feature, and NaN or inf
    in a used row are refused;
  - a dataset with a different fingerprint, a tampered dataset (fingerprint does not verify) and rows out of
    key order are refused;
  - the input dataset is unchanged after fit and predict (content hash before and after).
- Prediction keeps the order and keys of `prediction_indices`, including a shuffled index list.
- Artifacts: save and load round trip predicts identically (exact equality: the numbers are stored exactly);
  edited metadata, edited coefficients, an unknown schema version and an existing destination are refused; a
  failed publish leaves nothing behind.
- Reproducibility: the same inputs give the same `model_id` in a separate process.
- Integration: build the synthetic dataset (PR #11 fixture), train each model on the first part, predict on
  the rest, save, load, predict again.
- Every new test mutation-checked with `PYTHONDONTWRITEBYTECODE=1`; full suite at the end.

### Acceptance

The external plan's checklist, with two substitutions: `mean` instead of `DummyRegressor`, and no tree model
(deferred to a dependency review). The serialization trust boundary is documented as "no code is loaded".

### PR #12 plan review (external, 2026-11-12): revisions

The external final plan accepts the five differences above and adds requirements. Checked against the code and
statsmodels 0.15.0, numpy 2.5.3:

- **Training indices strictly increasing** (refused otherwise, never sorted); prediction indices in any order,
  output in the caller's order.
- **Labels attached after prediction.** The estimator receives only the feature matrix; the `label` column of the
  output is joined afterwards, for evaluation only.
- **One shared dataset verification.** `dataset.py` gains `verify_dataset(data, metadata, fingerprint)`, the check
  `read_dataset` already does (content hash, then fingerprint), and `read_dataset` calls it. Training and
  prediction call it on every call, for an in-memory `build_dataset` result and a loaded one alike; no fingerprint
  logic in `training.py`. It also checks that the feature columns equal the metadata's ordered features.
- **ElasticNet, verified.**
  - Objective of `OLS.fit_regularized(method="elastic_net")`: `0.5 * RSS / n + alpha * ((1 - L1_wt) * |b|_2^2 / 2
    + L1_wt * |b|_1)` (glmnet scale).
  - `alpha` may be a vector, one weight per coefficient: the intercept column gets weight 0, so it is not
    penalized.
  - The result exposes `converged`; a fit that did not converge is refused. Solver settings are fixed constants
    (`maxiter`, `cnvrg_tol`), recorded in the artifact and part of `model_id`.
  - Two identical fits give bit-identical coefficients (checked).
- **Ridge on the same alpha scale.** Ridge minimizes `0.5 * RSS / n + alpha * |b|_2^2 / 2`, that is
  `solve(Xs'Xs + n * alpha * I, Xs'yc)` with `np.linalg.solve`, so `alpha` means the same thing in both models.
  Independent test: it equals statsmodels `fit_regularized` with `L1_wt = 0` and a zero intercept weight.
- **Seed.** Recorded and part of the identity; the current models are deterministic, so changing it does not
  change coefficients (stated in the docstring and tested).
- **Artifact.** `model.json` carries `artifact_schema_version = "model/1"`; malformed JSON, an unknown version,
  mismatched shapes, non-finite numbers and a mismatched `model_id` are refused. Tests split into
  `test_ml_models.py`, `test_ml_training.py`, `test_ml_artifacts.py`.
- **Publisher.** Extracted to `ml/publish.py` only as a move (same lock, same atomic rename); every PR #11 test
  must pass unchanged. If the move turns out to need behaviour changes, only the lock primitive is shared.
- **Cost to note.** Verifying the dataset recomputes the content hash over every row on each fit and predict.
  On the synthetic dataset this is negligible; on the real research dataset it will be measured. If PR #13
  finds it too slow, a verified-dataset handle is designed there, not by weakening the check here.

### Result, PR #12 (2026-11-12)

Branch `ml-baseline-models` from `f06945b`. Full suite: 754 passed (689 before, 65 new).

- **`contracts.py`.** `ModelSpec(model, params, seed)`, frozen, validated, canonical; parameters are stored as
  floats (`1` and `1.0` give one hash). `REFERENCE_MODELS = ("mean",)`; `MODELS` and `ExperimentSpec` unchanged.
- **`models.py`.** `mean`, `ridge` (closed form, glmnet alpha scale), `elasticnet` (statsmodels, intercept weight 0,
  `maxiter = 1000`, `cnvrg_tol = 1e-8`, non-convergence refused). Standardization from the training rows only;
  a constant feature gets scale 1 and coefficient 0. `FittedModel` holds only finite Python floats and
  round-trips exactly through `float.hex`.
  - Found while testing: a BLAS matrix product gave predictions that differed in the last bit with the batch and
    memory layout. Inputs are now made C-contiguous and predictions summed feature by feature, so a row's
    prediction is exact and independent of its batch (tested).
- **`training.py`.**
  - `open_dataset` takes a `DatasetResult` or the `read_dataset` pair and verifies it via
    `dataset.verify_dataset`. It also checks the column layout, float dtypes, unique keys and (`date`, `symbol`)
    order.
  - `fit_model` (strictly increasing indices) and `predict_model` (any order, kept) check every index rule.
  - Output columns: `row`, `date`, `symbol`, `label_name`, `prediction`, then `label` (attached after
    prediction), `model_id`, `dataset_fingerprint`.
  - Training metadata: rows, ordered keys hash, first and last dates, first `label_start`, last `label_end`.
  - `model_id` covers the spec, the exact fitted numbers and the metadata, including the dataset provenance,
    the solver settings, `provenance.environment()` and a hash of the ML code files.
- **`artifacts.py`.** One `model.json` (`model/1`), published with `ml/publish.py`, verified by reloading before it
  becomes visible, never overwritten. `load_model` revalidates the spec, the fitted shapes and values, and
  recomputes `model_id`. No pickle or joblib: loading runs no code.
- **`publish.py`.** The lock and atomic publish moved out of `dataset.py` unchanged; `dataset.verify_dataset` is
  now the one integrity check (`read_dataset` calls it). All 27 PR #11 dataset tests pass unchanged.
- **Tests.** `test_ml_models.py` (17), `test_ml_training.py` (25), `test_ml_artifacts.py` (23):
  - ridge equals statsmodels with `L1_wt = 0` and an augmented least-squares solution (to 1e-10), and its
    predictions equal the independent solution;
  - lasso zeroes a noise feature; the intercept of both models is never shrunk;
  - non-convergence is refused; bad indices and bad inputs are refused;
  - leakage: prediction labels changed, or rows outside `train_indices` changed, give an identical fit and
    predictions;
  - a tampered dataset, a foreign dataset and self-consistent malformed datasets are refused: a label copy or a
    key column among the features, a renamed feature, an object dtype, unsorted rows, a duplicate key;
  - edited artifacts are refused (16 edits);
  - the same `model_id` comes out of an in-memory dataset, a loaded dataset and a separate process.
- **Mutation check.** 46 mutations over all new code and the moved publisher, all caught. Three survived the
  first run (predict ignoring scales, the fitted float type check, the label read in prediction) and were caught
  after the tests were strengthened.
- **Limitations.**
  - Verifying the dataset recomputes its content hash on every call: about 5.6 s for 500,000 rows and 15
    columns. PR #13 decides whether it needs a verified-dataset handle.
  - The real `research.duckdb` still predates the PR #10 build metadata and is refused until the next
    `quant build`, so no real-data run yet.
  - Bit-identical results are guaranteed on this machine and environment only; the environment is recorded in
    every model.
  - No tree model; it needs a dependency review.

## PR #12 review (external, 2026-11-12, head `8ed4341`): assessment and plan

No P1 correctness defect was found. Each point was checked against the code.

1. **`check_trained` does not validate the metadata structure: confirmed.**
   - A hand-built `TrainedModel` whose metadata is malformed but whose `model_id` is recomputed passes
     `check_trained`.
   - With `features` as plain names or an integer `label`, prediction is refused only later, by the
     feature comparison.
   - With no `training` summary, prediction passes and the model can be saved, so an artifact without its
     training interval summary (which PR #13 needs) is possible.
   - Plan: one `check_metadata(metadata, n_features)`, run in `TrainedModel.__post_init__`, so an invalid
     trained model cannot exist, and again in `check_trained` (save, predict, load).
   - Exact keys: `label`, `horizon`, `features`, `dataset_fingerprint`, `dataset_provenance`, `training`,
     `solver`, `environment`, `model_code_hash`.
   - `features`: a list of `[name, source_version]` string pairs with unique names, one per coefficient, equal
     to the provenance's `feature_versions` and to its `columns` layout.
   - `label` and `horizon` (an int > 0) equal those of the provenance.
   - `dataset_fingerprint` equals `dataset.fingerprint(provenance, provenance["content_hash"])`, so the copied
     provenance is bound to the fingerprint.
   - `training`: exact keys; `rows` an int > 0; `keys_hash` 16 hex characters; ISO dates with
     `first_date <= last_date`, `first_label_start > first_date` and `last_label_end > last_date`.
   - `solver` a mapping; `environment` a string; `model_code_hash` 16 hex characters.
   - Each malformed case gets a test (direct construction and edited artifact); `check_trained` still recomputes
     `model_id` because the nested metadata stays mutable.
2. **`FittedModel` does not validate its model name: confirmed.** It will be checked against
   `MODELS + REFERENCE_MODELS`, with a test.
3. **Features from `cols[2:-3]`: accepted as is, recorded.** The full layout is checked right after, so a schema
   change fails closed rather than being misread. An explicit ordered feature list in the dataset metadata would
   change the dataset builder's metadata and every dataset fingerprint, which PR #12 does not touch. It is a
   follow-up for the next dataset schema version.
4. **`code_hash` scope: clarified by renaming.** It is meant to cover the code that determines the fitted numbers
   and the predictions (canonical, contracts, dataset verification, models, training). `artifacts.py` and
   `publish.py` only store and move bytes, and the storage format is versioned separately by
   `artifact_schema_version`; adding them would change `model_id` for storage-only edits. The metadata key
   becomes `model_code_hash`, with a docstring that states this scope. No artifact exists outside tests, and the
   schema stays `model/1`.
5. **Verification cost: follow-up for PR #13**, as already recorded (measure on the walk-forward workload; any
   verified-dataset handle must not skip verification).

CI: `gh` is not installed here; the user confirms the run on the pushed head.

### PR #12 review fixes: result (2026-11-12)

Full suite: 786 passed (754 before). ML model tests: `test_ml_models.py` 17, `test_ml_training.py` 53,
`test_ml_artifacts.py` 27.

- **Metadata contract.** `training.check_metadata` runs in `TrainedModel.__post_init__` and again in `check_trained`
  (save, load, predict), so an invalid trained model cannot be built, even with a recomputed `model_id`.
  - It checks every rule of the plan: exact keys; feature pairs, label and horizon equal to the provenance;
    `dataset_fingerprint` reproduced from the copied provenance; a well-formed, ordered training summary
    (`rows` within `1..kept_rows`); `solver`, `environment` and `model_code_hash`.
  - `model_id` must be 16 hex characters.
  - Tests:
    - 24 malformed-metadata cases are refused at construction;
    - metadata edited after construction is refused on predict and save, including a value that cannot even be
      hashed;
    - edited artifacts are refused (label, provenance, code hash, fitted model name).
- **`FittedModel`** refuses a model name outside `MODELS + REFERENCE_MODELS`.
- **`model_code_hash`** (renamed from `code_hash`) with `MODEL_CODE_FILES`: the code that determines fitted numbers
  and predictions. Storage (`artifacts.py`, `publish.py`) is versioned by `artifact_schema_version`, not by this
  hash.
- **Not changed, recorded:**
  - feature discovery by position, followed by the full layout check, until the next dataset schema version;
  - the verification cost, which is a PR #13 follow-up.
- **Mutation check.** 66 mutations, all caught: the 46 earlier ones (re-anchored) and 20 new ones, one per
  metadata rule plus the structure and model-name checks. Two survived the first run (the structure re-check in
  `check_trained`, the `model_id` format) and were caught after two tests were added.

## PR #12 second review (external, 2026-11-12, head `fa0a111`): assessment and plan

One P2 point; the earlier findings are recorded as addressed or accepted follow-ups.

- **`open_dataset` accepts any float dtype, then converts to float64: confirmed.** A resealed (self-consistent)
  dataset with a `float32` feature column, or a `float32` label, trains today. The dataset builder always
  produces `float64`, so this only matters for a dataset produced some other way. It still breaks the "nothing
  is coerced" contract, because `float32` precision is already lost and widening hides that.
- **Plan.**
  - Require exactly `float64` for every feature column and the label in `open_dataset`, before any conversion,
    with the message `column <c> has dtype <d>, expected float64`.
  - Tests on resealed datasets: a `float32` feature, a `float32` label and a `float16` feature are refused; the
    existing `object` dtype case stays.
  - Update the mutation for this check (`kind == "f"` instead of exact `float64` must be caught).

### PR #12 second review fix: result (2026-11-12)

Full suite: 786 passed (the new cases extend an existing test).

- `open_dataset` requires exactly `float64` for every feature and the label, before any conversion.
- Resealed datasets with a `float32` feature, a `float16` feature, a `float32` label or an `object` feature are
  refused with `expected float64`.
- Mutation check: 67 mutations, all caught. The dtype mutation was re-anchored, and a new one restoring the old
  `kind == "f"` rule is caught.

## PR #12 third review (external, 2026-11-12, head `0745232`): assessment and plan

- **`FittedModel` is not deeply immutable: confirmed.**
  - Built with lists, it keeps the caller's lists. Mutating `coef` afterwards changes `models.predict` (1.0
    became 999.0 in a check).
  - Through `TrainedModel` the edit would be caught, because `check_trained` recomputes `model_id`, but
    `FittedModel` is its own contract.
- **Plan.**
  - `FittedModel.__post_init__` copies `means`, `scales` and `coef` into tuples before validating them. A
    non-sequence (a string, a mapping, a scalar) is refused, and the caller's list is never kept, as
    `contracts.freeze` does for parameters.
  - Tests:
    - list inputs become tuples;
    - mutating the caller's list after construction changes nothing;
    - the fields cannot be mutated;
    - `to_json` and `predict` work on the normalized state;
    - a string vector is refused.
  - Add a mutation (no tuple copy) to the PR #12 set.
- **PR description.** It still reports 754 tests and 65 mutations; the corrected text is given to the user to
  paste (no `gh` here).

### PR #12 third review fix: result (2026-11-12)

Full suite: 787 passed (786 before).

- `FittedModel.__post_init__` copies `means`, `scales` and `coef` into tuples before validating them; anything
  that is not a list or tuple (a string, a mapping, a scalar, a numpy array) is refused.
- Test (`test_fitted_state_is_deeply_immutable`):
  - list inputs become tuples;
  - mutating the caller's lists afterwards changes neither the state nor the prediction, which is checked
    against a hand computation;
  - the fields cannot be edited or reassigned;
  - the JSON round trip is exact;
  - non-sequences are refused.
- Mutation check: 69 mutations, all caught (2 new: no tuple copy, no sequence type check).

## PR #13: walk-forward runner (plan, 2026-11-12)

Input: the external PR #13 plan. Branch `ml-walk-forward` from `main` at `b840980` (PR #12 merged). This
section maps that plan onto the code and records the points it leaves open.

### Audit

- **Splits.** `folds(calendar, split)`, `purge(train, fold)`, `check_fold(train, test, fold, split)` and
  `session_offset` exist and are tested. Embargo is `embargo_sessions` before the test start; training labels end
  strictly before it.
- **Models.** `fit_model` / `predict_model` work on row positions of one verified dataset (PR #12).
  `ExperimentSpec(model, params, seed)` maps to `ModelSpec(model, params, seed)`; it accepts only `ridge` and
  `elasticnet`.
- **Run lifecycle.**
  - `ResultsStore.start_run(run_id, kind, period, params: ResearchParams, now, pattern)` stores
    `stored_params(params)` plus provenance; it has no slot for an ML experiment configuration.
  - `finish_run` sets `ok`; an exception sets `failed`; runs are never deleted (`invalidate`).
  - `log_hypothesis` stores a NaN or None p-value as NULL, which keeps it on record but outside the FDR family.
  - `hypothesis_q` counts only tests of `ok` runs.
  - Every logged engine refuses a second run of the same configuration until the earlier one is invalidated
    (`already_run` in interactions and portfolio).
- **Rank IC convention.** `interactions.daily_ic`: Spearman as Pearson on average ranks, at least 30 stocks per
  date (`MIN_STOCKS = 30`). `stats.mean_test(series, lag)`: Newey-West, `None` when fewer than 3 values or the
  standard error is undefined.
- **Baseline.**
  - `BaselineSpec(factor, source_version, field)` names a Factor Engine score in `rs.stock_factors`. The ML
    dataset does not contain it unless it is also a model feature.
  - Factors carry a sign note, not an expected-return orientation; `reversal_1w`, for example, is "high =
    recent loser".
- **Fixtures.** The PR #11 synthetic warehouse has 4 stocks: its dataset can never reach 30 stocks per date.
- **Real data.** `research.duckdb` still predates the PR #10 build metadata, so ML refuses it until the next
  `quant build` (run by the user).

### Points the external plan leaves open (to approve)

1. **Where the baseline score comes from.** Read the `BaselineSpec` field from `rs.stock_factors`, read-only, for
   the test rows' exact (`date`, `symbol`) keys. Checks:
   - the dataset's `build_id` must be the build in `research.duckdb` (`check_build` as for the dataset);
   - `BaselineSpec.source_version` must equal the dataset's recorded `stock_factors` source version, so model and
     baseline come from one build.
   - A missing baseline value removes the row from both model and baseline IC (the intersection rule of the
     external plan).
2. **Baseline orientation (a decision).** "Model IC minus baseline IC" is unfair whenever the factor's IC is
   negative: the model learns its signs, while the baseline would be scored as if high meant better. Options:
   - **(A, recommended)** In each fold, the baseline's sign is the sign of its mean daily rank IC on that fold's
     purged training rows. It is learned with the same information the model has and never from test labels,
     then frozen for the test window and recorded in the fold metrics.
   - (B) A sign declared in the experiment. This adds a field to `BaselineSpec` and changes `experiment_id`;
     declaring it without looking at data is hard, and research-period evidence would be chosen with
     hindsight.
   - (C) No orientation: the comparison is biased for any negatively signed factor.
3. **Validity decided before any fit.** Which rows are usable depends only on the dataset's keys and the
   baseline's presence; predictions and dataset labels are always finite. So the stock count per date and the
   valid dates per fold are computed before fitting.
   - If any fold has fewer than 60 valid dates, or `test_sessions < 60`, or the test windows overlap
     (`step_sessions < test_sessions`), the experiment is refused before a run starts and nothing is logged.
   - Dates with fewer than 30 stocks are dropped and counted.
   - A date where model predictions or baseline scores are constant has no defined IC. It is dropped for both
     and counted (`undefined_ic_dates`), because this depends on features only. If drops then leave a fold
     below 60 valid dates, the run fails; it is never re-cut.
4. **One run per experiment and dataset.** As in the other engines, a second run of the same `experiment_id` on
   the same dataset fingerprint is refused while an earlier `ok` or `running` run exists; invalidate it first.
   Changing parameters after a result is a new experiment and a new logged test, visible in `quant audit fdr`.
5. **Storing the experiment in the run.** `start_run` gains an optional, additive `config: dict | None`, stored
   next to the parameters; existing runs and callers are unchanged. ML runs store `ExperimentSpec` (canonical),
   `experiment_id`, the dataset fingerprint and the split version. `kind = "ml"`, `period = "research"`,
   pattern `(experiment_id, model_code_hash)`.
6. **Order of persistence.**
   - `start_run`, then the primary label is logged with a NULL p-value (declared before any result exists).
   - Folds are computed into a temporary run directory: fold models via `save_model`, `predictions.parquet`
     and `metrics.json` per fold.
   - Then the primary test is computed, `manifest.json` written, and the directory published atomically
     (`ml/publish.py`) at `data/ml/runs/<run_id>`.
   - Then the p-value is logged and `finish_run` called.
   - The manifest holds no status: `research_runs` is the only source of truth, so a published directory of a
     run that later failed is never mistaken for a valid run.
   - An exception leaves the run `failed` with its test outside the FDR family, as for every engine.
7. **Fixtures.**
   - A synthetic warehouse with at least 40 stocks and enough sessions for real folds, built through the normal
     pipeline (factors included) for the integration test.
   - Hand-made, resealed datasets for the edge cases (29 or 30 stocks, 59 or 60 dates, ties, constant vectors).
   - The 30 and 60 thresholds are constants of the primary-test definition, recorded in the run.

### Design

- `ml/evaluation.py`, pure functions:
  - `rank_ic(x, y)`: average ranks, then Pearson; NaN when either side is constant.
  - `baseline_sign(train_rows, baseline, labels)`;
  - `daily_incremental_ic(predictions, baseline, labels, sign)`: identical samples, counts of dropped dates;
  - `primary_test(series, horizon)`: `stats.mean_test(series, horizon - 1)`.
- `ml/runner.py`:
  - `plan_folds(dataset, calendar, split, baseline)`: fold rows, purge, `check_fold` on the exact frames, and
    the validity counts, all before any fit;
  - `run_experiment(store, spec, ...)`: lifecycle, duplicate guard, per-fold fit and predict (labels joined only
    after prediction), artifacts, primary test.
- CLI: `uv run quant ml run <experiment.json>`. The JSON names everything (dataset features, label, period,
  split, model, params, seed, baseline, hypothesis); nothing is defaulted. It prints the run id, status, fold
  count, primary label and artifact path.
- Verification cost: count the full dataset verifications and time them in the integration test, then report
  them. The cost is about 2 verifications per fold; no fast path is added in this PR.
- No change to the Factor Engine, the Risk Engine, econ, portfolio, backtest, the warehouse schema, or the
  validation and holdout gates.

### Tests (summary)

- Fold planning:
  - expanding and rolling folds; overlapping test windows refused;
  - purge with a suspended symbol; rows just before, on and after `embargo_start`;
  - `check_fold` called on the exact frames;
  - the cutoff rule;
  - `test_sessions < 60` refused.
- Leakage:
  - test labels mutated give identical predictions and baseline signs;
  - rows outside the fold's training set mutated give an identical fit;
  - the baseline sign uses training rows only;
  - model and baseline use the same keys on every date, including when a baseline value is missing.
- Statistics:
  - hand-computed Spearman with ties;
  - a constant vector is undefined and counted;
  - 29 versus 30 stocks; 59 versus 60 dates;
  - incremental IC equals model minus signed baseline;
  - the Newey-West lag is `horizon - 1`;
  - exactly one hypothesis row per run.
- Lifecycle:
  - the label is logged with NULL before the result;
  - a failure leaves the run `failed` and outside FDR;
  - the duplicate run is refused;
  - an invalidated run allows a rerun;
  - atomic run directory, never overwritten;
  - reproducible fold boundaries, key hashes, model ids and predictions.
- Integration: a full run on the larger synthetic warehouse, with the verification count and time.
- Mutation check on every guard; full suite.

### PR #13: approved decisions (2026-11-12)

The user approved the plan with option A and these refinements; they override the points above where they differ.

1. **Baseline direction (A).**
   - For each fold, the daily Spearman IC of the baseline score against the label is computed on the fold's
     purged training rows: dates with at least 30 stocks that have a finite baseline, and a defined IC.
   - Their mean over dates (each date weighs the same) gives the sign: > 0 keeps the score, < 0 negates it.
   - A mean of exactly 0, or fewer than 60 valid training dates, fails the fold; no sign is chosen by default.
   - The sign is fixed for the fold's whole test window. Fold metrics record the raw mean IC, the sign and the
     number of valid training dates.
   - Test labels are never used for it.
2. **Preflight, before any run or hypothesis row.**
   - Checks:
     - unique keys;
     - non-overlapping test windows (`step_sessions >= test_sessions`);
     - `test_sessions >= 60`;
     - baseline coverage;
     - stocks per test date (at least 30 with a baseline);
     - valid test dates per fold (at least 60);
     - every fold's `check_fold` on the exact purged frames;
     - the baseline sign rule above.
   - Any failure rejects the whole experiment.
   - The preflight reads training labels (for the sign) and test keys only, never test labels.
3. **Idempotency: one research result per experiment and dataset.**
   - Runs are identified by `pattern_name = experiment_id` and `pattern_version = dataset fingerprint`.
   - A new run is refused when an earlier run of the pair:
     - is `ok` (a result exists);
     - is `running` (in progress, or killed: inspect, then invalidate);
     - has a non-NULL primary p-value logged;
     - or has a published run directory (a result was computed).
   - A run that `failed` before any result (NULL p-value, no directory) allows a retry. Its hypothesis row stays
     NULL, and only `ok` runs enter `hypothesis_q`, so a retry adds no test to the FDR family.
   - Invalidation (with a reason, for an established defect) remains the only way past a result.
   - A different configuration, baseline policy or method is a new `experiment_id` and a new test in the global
     FDR family. The baseline policy and the thresholds are recorded in the run's configuration.
4. **Run lifecycle.**
   - `start_run(..., config=None)` is backward compatible: without `config` the stored row and config hash are
     unchanged.
   - `research_runs` is the only source of truth for status; the manifest has no status.
   - Order:
     - start the run;
     - log the primary label with NULL;
     - compute every fold into a temporary directory;
     - publish the run directory atomically;
     - then, in one transaction, log the p-value and `finish_run`.
   - A crash at any step leaves either no directory, or a directory with the run not `ok`, which the
     idempotency rule refuses to rerun. A run is never `ok` without its directory, and the label is one
     `(run_id, label)` row, so it is never logged twice.

### Result, PR #13 (2026-11-12)

Branch `ml-walk-forward` from `b840980`. Full suite: 827 passed (787 before, 40 new: `test_ml_evaluation.py` 7,
`test_ml_runner.py` 19, `test_ml_experiment_file.py` 14). The suite now takes about 3 minutes (2 before), because
the runner tests build a wider warehouse and run complete experiments.

- **`ml/evaluation.py`.**
  - `rank_ic` (average ranks, Pearson; NaN when constant) and `eligible` (30 stocks with a finite baseline per
    date).
  - `baseline_sign` (mean of daily training ICs, each date weighted equally; a mean of 0 or fewer than 60 dates
    fails).
  - `daily_incremental_ic` (identical rows for model and baseline, dropped dates counted).
  - `primary_test` (`stats.mean_test`, lag `horizon - 1`, chronological and unique dates, undefined fails).
- **`ml/runner.py`.**
  - `load_calendar` and `load_baseline` query only the dataset period.
  - `preflight` builds the folds and checks every refusal of the approved decisions before any write.
  - `previous_runs` applies the idempotency rule.
  - `run_experiment` runs the approved lifecycle: start, NULL label, folds into a temporary directory, atomic
    publish (`verify_run` reloads every fold model), then p-value and `finish_run` in one transaction.
  - `check_build` covers the baseline.
- **`ml/experiment_file.py`, `ml/cli.py`.** `quant ml run EXPERIMENT.json` takes a file with exact keys and no
  defaults. Refusals exit with code 2 and log nothing.
- **`results.py`.** `start_run(..., config=None)`: additive; without `config` the stored row and config hash are
  unchanged (tested).
- **Run artifacts** (`<data dir>/ml/runs/<run_id>/`):
  - `manifest.json` (no status): experiment, ids, dataset metadata, fold metrics, primary result, verification
    count;
  - per fold: `model/model.json`, `predictions.parquet` (with `baseline_score` and `baseline_sign`),
    `daily_ic.parquet`, `metrics.json`.
- **Verification cost.**
  - One run on the wide synthetic warehouse (45 stocks, 520 sessions, about 20,000 dataset rows, 6 folds) does
    13 full dataset verifications (1 + 2 per fold). The folds take about 2.2 s and the whole run about 3 s.
  - From the PR #12 measurement (5.6 s per 500,000 rows), a real research dataset with 25 folds would spend
    about 5 minutes on verification. That is acceptable for a run that happens once per experiment, so no
    verified-dataset handle is added.
- **Tests.**
  - The real run logs one test, finishes `ok`, and enters `hypothesis_q` once. Each fold model is fitted on
    exactly its purged training rows (row count and key hash), and test keys never repeat across folds.
  - `check_fold` receives the exact frames.
  - Mutating this fold's and later test labels changes neither the earlier folds' baseline signs nor this
    fold's fit and predictions.
  - Preflight refusals log nothing: overlapping windows, `test_sessions` of 59, the training row minimum, too few
    stocks, a zero mean sign, a test row tested twice.
  - Idempotency: a rerun is refused; a different seed is a new experiment; after invalidation a rerun is
    allowed and the FDR family counts 2 tests.
  - The label is NULL and the run `running` while the result is computed.
  - A technical failure before the result allows a retry that adds one FDR test only. A crash after publishing
    leaves the run `failed`, rolls back the p-value and blocks a rerun. A killed `running` run or a logged
    p-value blocks a rerun.
  - A fold left below 60 dates after prediction fails the run.
  - Reproducibility across two stores: manifest, predictions and primary result are identical.
  - `verify_run` detects a damaged model or incomplete predictions.
  - The CLI runs once, then refuses.
- **Mutation check.** 31 mutations over the evaluation, runner, idempotency, lifecycle, artifact and `start_run`
  code, all caught. Three survived the first run (two refusals whose tests matched the wrong message, and the
  post-prediction date minimum) and were caught after the tests were made specific.
- **Hard constraints kept.**
  - Every query is bounded by the dataset period, which is at most 2023-12-31.
  - No validation or holdout read.
  - Point-in-time data through the PR #11 builder only.
  - Fit on purged training rows only.
  - Test labels are read only from `predict_model` output, after prediction. The preflight reads training labels
    (for the baseline sign) and test keys and baseline scores only.
- **Limitations.**
  - No real-data run yet: `research.duckdb` must be rebuilt with `quant build` first.
  - A killed run stays `running` and blocks its experiment until it is inspected and invalidated, by design.
  - The synthetic data has no signal, so the test checks mechanics, not performance.

## PR #13 review (external, 2026-11-12, head `40d4185`): assessment and plan

Each point was checked against the code.

1. **[P1] `verify_run` checks row counts only: confirmed.**
   - A prediction or daily-IC file with the same row count but changed values or keys passes, and PR #14 will
     read these files.
   - Plan, two layers:
     - **Integrity:** the manifest records a SHA-256 of every fold file (model, predictions, daily IC,
       metrics), and `verify_run` recomputes them. This detects any accidental or deliberate change to a file.
       It is not a signature: whoever rewrites the files and the manifest can recompute the hashes.
     - **Consistency:** `verify_run` recomputes from the stored predictions:
       - the prediction keys are unique within and across folds, and equal the recorded test-key hash;
       - one model id and one baseline sign per fold, equal to the metrics;
       - the daily IC, recomputed from predictions, baseline, sign and label, equals `daily_ic.parquet`
         exactly;
       - the fold means equal the metrics;
       - the primary test, recomputed from the concatenated daily IC, equals the manifest's.
   - Optional `verify_run(path, dataset)`: with the verified dataset, every stored prediction is also
     recomputed from its loaded model (`predict_model`) and must be identical. PR #14 can use it.
   - Tests: changing one prediction value, one label, one key, one daily-IC value or the primary p-value, each
     with the row count kept, is refused; the recompute-from-model check catches a prediction file regenerated
     with consistent hashes.
2. **[P1, conditional] Finite labels: the invariant holds, and it will be made explicit.**
   - `01_daily_panel.sql` drops rows with any price <= 0 or `adj_ratio <= 0`, so `adj_open > 0` and the labels
     (`close / open - 1`) are finite whenever they are not NULL.
   - A DuckDB `DOUBLE` divided by 0 gives `inf` (checked), so this invariant is what keeps them finite. NULL
     labels are excluded by `label_null`.
   - Plan:
     - every IC sample is the finite paired sample: rows where the score, the baseline and the label (and,
       after prediction, the prediction) are all finite;
     - the 30-stock rule applies to that paired sample;
     - in the preflight (test labels not read) the rule applies to finite baselines, and after prediction it
       is applied again to the full paired sample. A date that falls below 30 is counted as
       `too_few_stocks_dates`.
   - Tests:
     - NaN, `+inf` and `-inf` in the label, the baseline and the prediction;
     - 29 versus 30 valid pairs;
     - a dataset-boundary test: every kept label and feature of a built dataset is finite, including the
       synthetic zero-price row.
3. **[P2] Concurrent duplicate runs: partly confirmed.**
   - Two processes cannot write `results.duckdb` at the same time (single writer), so they are serialized.
   - Two connections in one process can both pass the pre-check, though.
   - Plan: an `fcntl` lock (`ml/publish.acquire`) keyed by (`experiment_id`, dataset fingerprint), in an
     `ml-locks` directory next to the results database. It is held from the pre-check to `finish_run`.
   - A held lock refuses the run before anything is logged; the pre-check stays for its message.
   - Tests: a subprocess holding the lock refuses the run and nothing is logged; after the holder dies, the run
     proceeds.
4. **[P2] Duplicate (`date`, `symbol`) rows in evaluation: confirmed.** The evaluation frames carry `symbol`, and
   `daily_incremental_ic` / `daily_baseline_ic` refuse duplicate keys. Regression test included.
5. **[P2] Eligible versus valid dates: confirmed as a naming gap.**
   - Fold metrics will define:
     - `test_eligible_dates`: preflight, finite baseline and at least 30 stocks;
     - `valid_dates`: after prediction, the full finite paired sample with both ICs defined;
     - `undefined_ic_dates` and `too_few_stocks_dates`: the post-prediction drops.
   - The manifest carries these definitions.
   - Test: a run with 70-session test windows and constant predictions on 3 dates per fold finishes `ok` with
     `test_eligible_dates = 70 > valid_dates = 67` and `undefined_ic_dates = 3`.

CI: run #51 was in progress at review time; `gh` is not available here, so the user confirms it.

### PR #13 review fixes: result (2026-11-12)

Full suite: 848 passed (827 before). `test_ml_evaluation.py` 18, `test_ml_runner.py` 28, one new test in
`test_ml_training.py`.

- **Artifact verification** (`ml/run_verify.py`, `verify_run(path, dataset=None)`):
  - Integrity: the manifest has a `files` map, the SHA-256 of every fold file (lock files excluded), and every
    hash is recomputed.
  - Consistency, recomputed from the stored predictions:
    - row count; one fold, model, sign and dataset per fold;
    - test keys unique within and across folds, and equal to the recorded key hash;
    - daily IC equal to `daily_ic.parquet`; counts and means equal to the fold metrics;
    - the primary test equal to the manifest's.
  - With the dataset, every prediction and label is also recomputed from the loaded fold model.
  - The run's publish step verifies without the dataset: those predictions were just computed by the same code.
  - Tests, each change keeping the row count:
    - a sign flip of the predictions, permuted labels, a changed key, the sign, a daily-IC value, a dropped row,
      a key repeated from another fold, an edited fold mean, an edited primary p-value, and another fold's
      valid model in place of the right one are each refused. The file changes are caught by the hashes and,
      after the hashes are rewritten, by the consistency layer.
    - A coherent forgery (predictions cubed, so the ranks, daily IC, metrics, primary result and hashes all stay
      consistent) passes without the dataset and is refused with it. Rank statistics cannot see a monotone
      change; only the model recomputation can.
- **Finite paired samples** (`evaluation.eligible(frame, columns)`):
  - The 30-stock rule counts rows whose used values are all finite: baseline only in the preflight;
    baseline and label for the sign; prediction, baseline and label for the test.
  - The upstream invariant (prices <= 0 dropped by `01_daily_panel.sql`, so labels are finite) is documented
    and tested at the dataset boundary.
  - Tests: NaN, `+inf` and `-inf` in each column; 29 versus 30 finite pairs, for the test sample and the sign
    sample.
- **Duplicate keys.** Evaluation frames carry `symbol`; duplicate (`date`, `symbol`) rows are refused.
- **Concurrent runs.** An `fcntl` lock per (`experiment_id`, dataset fingerprint) in `ml-locks/` next to the
  results database is held from the pre-check to `finish_run`. Test: while a subprocess holds it, the run is
  refused and nothing is logged; once the holder is killed, the run finishes `ok`.
- **Eligible versus valid dates.** `evaluation.METRIC_DEFINITIONS` is stored in the run configuration and the
  manifest. Test: 70-session windows with constant predictions on 3 dates per fold finish `ok` with
  `test_eligible_dates = 70`, `valid_dates = 67` and `undefined_ic_dates = 3`.
- **Mutation check.** 45 mutations, all caught: the 31 earlier ones (re-anchored, with one retired: the old
  NaN-baseline rule, now part of the finite-pair rule) and 15 new ones. Six survived the first run (the sign
  sample, and five verification checks whose tests matched another check's message or were missing) and were
  caught after the tests were made specific.

## PR #13 second review (external, 2026-11-12, head `02a1407`): assessment and plan

1. **[P2] `metrics.json` is not compared with the manifest: confirmed.** `verify_run` checks the fold metrics in
   `manifest["folds"]` only. A `metrics.json` rewritten together with its hash would pass, so the two sources
   could disagree.
   - Plan: each fold's `metrics.json` must equal its manifest entry, compared as JSON.
   - Test: an edited `metrics.json` with its hash rewritten is refused.
2. **[P2] The dataset-backed check covers predictions and labels only: confirmed.**
   - Row, date and symbol are not compared with the dataset, and the baseline cannot be checked at all, because
     it comes from `stock_factors`, not from the dataset.
   - Plan:
     - With a dataset, the stored `row`, `date`, `symbol`, `label` and `prediction` must all equal the
       dataset row and the recomputed prediction.
     - `verify_run(path, dataset, baseline)` also takes the baseline aligned to the dataset rows; the stored
       `baseline_score` must equal `baseline[row]` (NaN equal to NaN).
     - New `runner.reverify(store, spec, path)` rebuilds both inputs exactly as `run_experiment` does
       (`check_build` with the baseline, `build_dataset`, `load_baseline`) from `research.duckdb`. It refuses
       unless the spec's `experiment_id`, the dataset fingerprint and the build id equal the manifest's, then
       runs the full check.
     - The baseline is verified only against the factor build that the build check accepts. A rebuilt
       `research.duckdb` (a new build id) cannot re-verify an older run: this is stated in the contract.
   - Tests:
     - a changed date, symbol, row or baseline score, with hashes and metrics kept consistent, is refused with
       the inputs and passes without them;
     - `reverify` passes on an untouched run and refuses a wrong experiment.
3. **[P3] Integrity is not authenticity: agreed, documented.** The `run_verify` docstring and this plan will
   state three levels:
   - `verify_run(path)`: the run directory is internally consistent, and its files match its manifest;
   - with inputs (`reverify`): its predictions, keys, labels and baseline follow from the recorded model,
     dataset and factor build;
   - neither proves who wrote the directory: a complete, consistent replacement of the directory passes both.
   The authoritative record is `research_runs` and `hypothesis_log` (status, the logged p-value). PR #14 must
   compare the manifest's primary p-value with the logged one, and must not treat a verified directory as proof
   of origin. No schema change is made for this.

### PR #13 second review fixes: result (2026-11-12)

Full suite: 854 passed (848 before); `test_ml_runner.py` has 34 tests.

- **`metrics.json`.** Each fold's file must equal its manifest entry. Tests:
  - an edited manifest entry is refused;
  - the same edit in both sources (rehashed) is refused by the recomputation;
  - an edited `metrics.json` alone (rehashed) is refused.
- **Level 2, `verify_run(path, dataset, baseline)`.**
  - With the dataset, the dataset row named by each stored `row` must carry the stored date, symbol and label,
    and the recomputed prediction must equal the stored one.
  - With the baseline (aligned to the dataset rows), the stored `baseline_score` must equal it, NaN equal to
    NaN. A baseline without a dataset is refused.
  - A literal comparison of the returned `row` column with the requested rows was dropped: `predict_model`
    returns exactly the rows it is given, so the comparison could never fail. Row identity is the key, label and
    prediction check above.
- **`runner.reverify(store, spec, path)`.** It rebuilds the dataset and the baseline with `_inputs`, the same
  function `run_experiment` uses (`check_build` with the baseline, `build_dataset`, `load_baseline`). It refuses
  unless the experiment id, the dataset fingerprint and the build id equal the manifest's, then runs level 2.
- **Tests.**
  - Six coherent forgeries of one fold pass level 1 and are refused by level 2 and by `reverify`. Each one
    recomputes the daily IC, counts, means, key hash, `metrics.json`, the primary result and every hash. They
    are: predictions cubed (same ranks), a changed symbol, a changed date, shifted rows, reversed labels, and
    baseline scores halved (same ranks).
  - The halved baseline also passes with the dataset alone, which shows why the baseline input is needed.
  - `reverify` passes on the untouched run and refuses another experiment.
- **Verification levels** (`run_verify` docstring): (1) internal consistency; (2) consistency with the recorded
  inputs, while `research.duckdb` holds the run's build; (3) neither proves who wrote the directory.
  `research_runs` and `hypothesis_log` stay authoritative, and PR #14 compares the manifest's p-value with the
  logged one.
- **Mutation check.** 52 mutations, all caught (45 earlier, one re-anchored, and 7 new). One survived the first
  run (labels ignored at level 2) and was caught after a reversed-label forgery was added.

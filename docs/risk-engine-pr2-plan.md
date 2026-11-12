# Risk Engine PR 2: stress, construction grid, UI (plan, 2026-11-12)

Source: the external spec "Risk Engine PR 2 - Stress Testing, Construction Grid Integration & UI" (2026-11-12).
This document contains:
- the Phase 0 audit, done on `main` at `a46c661` after PR #8 was merged, with no code changed;
- the design decisions that need approval before any code is written.

## 1. Phase 0 audit: actual interfaces

### Risk Engine (PR 1, `src/quant_research/risk/`)

- `data.Snapshot(target, valuation_date, nav, cash, positions[symbol, shares, price, value], curves{name: (dates,
  equity)}, notes, reconciliation_gap)`.
  - Built by `paper_snapshot(con)` (both valuation curves required) and
    `backtest_snapshot(con, run_id, on)` (one stored curve).
- `data.market_inputs(con, symbols, on)`: ADV20, half-spread, sigma, ICB level 2 (current classification), last
  trade, stale sessions, at-floor flag. `data.clusters(con, symbols, on)`: the last month end on or before
  `on`. `data.exposure_returns(...)`: 120 sessions.
- `evaluate.assess(snap, inputs, rets, cluster_of, factor_z, policy, liq) -> Assessment(status, limit_results,
  metrics, positions, notes)` is pure. `evaluate.evaluate(store, snap, policy, now)` checks the period gate
  (research or forward only), registers the frozen policy, starts a run of kind `risk`, and stores the
  `risk_*` rows. `MODEL_VERSION = "risk-v1"`.
- `liquidity.position(...)` and `liquidity.liquidation_cost(...)` call `CostModel.side` (cost model v1) and the
  strategy's fee and tax. Invalid ADV gives an unknown result.
- `assess` returns per-position `beta`, `industry`, `cluster`, `weight`, `stale_sessions` and the liquidity
  fields. These are exactly the inputs a stress calculation needs, so stress needs no new loader.

### Portfolio construction (`src/quant_research/portfolio/`)

- `engine.run_targets(market, construction, score_fn, schedule, initial, industry, cash_on, costs, sell_lag,
  writedown_after) -> PortfolioResult` is pure.
  - It returns `equity`, `cash`, `n_names` and `trades` as (session, column, signed shares, price, cost), and
    persists nothing.
  - `writedown_after=60` gives the write-down equity curve, so a second valuation curve exists, as for the
    paper portfolio.
- `evaluate.evaluate(store, name, now)` stores `portfolio_results` (summary metrics only). The run parameters
  store `extra.config` (signal, schedule, construction, cash rule), the universe rule, the capitals,
  `n_control`, and provenance (commit, dirty flag, warehouse SHA, feature build, code hash).
  - It refuses a config that already has an `ok` run, and runs only on the research period.
  - The cost models (`COSTS`), the sell lag and the decision setting (1 bn VND, `v1_k1`) are code constants,
    identified only through the code hash.

### UI (`src/stock_ui/`)

- `app.py` defines `st.navigation` with 8 pages.
- Data access goes through `db.Reader.read(db, key, fn)`: read-only connections, a TTL cache, and stale
  results while the pipeline runs.
- `exposure_panel` recomputes the Exposure Engine live on the Backtest and Market watch pages.
- Risk runs are not shown anywhere in the UI yet.
- Tests in `tests/ui/test_pages.py` use the Streamlit `AppTest` harness on a synthetic data directory.

## 2. Answers to the audit questions (spec section 4.2)

1. **Can a construction run be rebuilt from its stored configuration?** The configuration yes: signal,
   schedule, construction, cash rule and universe are in `params.extra`. The cost model, the sell lag and the
   decision setting are code constants, recorded only through `code_hash`.
2. **Are all inputs identifiable?** Identifiable, but not recoverable.
   - The six valid runs (`20261009T171446-portfolio-C1..C6`) record commit `5114c19` with uncommitted
     changes (`git_dirty: true`), warehouse `592d0f8f...` and feature build `20261009T112210`.
   - The warehouse file of that day is not kept; today's is `8d4d550c...`.
   - The exact code is not recoverable either: the commit was dirty, and the code hash names files that are
     not stored.
3. **Can the run be reproduced?** No. A read-only re-simulation at the decision setting (1 bn VND, cost model
   v1, current code and data, on a copy of `results.duckdb`):

   | Config | CAGR stored -> now | other differences |
   |---|---|---|
   | C1 | identical (1e-16) | `avg_names` +0.28% (equity identical: a definition or bookkeeping change since the run, not explained) |
   | C2 | -0.17788 -> -0.17461 (+1.8%) | stale share -3.9%, average names +1.2% |
   | C3 | -0.06324 -> -0.06413 (-1.4%) | small |
   | C4 | identical (1e-16) | `avg_names` -0.27% |
   | C5 | -0.12703 -> -0.12770 (-0.5%) | stale share +0.7% |
   | C6 | -0.03548 -> -0.03488 (+1.7%) | industry weight -0.8% |

   The four configurations with an exit buffer (`exit_rank` 40) diverge. They rank deeper into the score list,
   so small changes in the rebuilt factor ranks or in the code since the run change which names are kept. No
   decision would change (all six stay far below equal weight), but these are not the recorded runs.
4. **Does a re-simulation need mutable current data?** Yes. `research.duckdb` is rebuilt from the warehouse,
   which grows every day and can be restated by the vendor.
5. **Can the engine give a result without persisting a construction experiment?** Yes, `run_targets` is pure.
   Only `portfolio.evaluate.evaluate` writes.
6. **Where would the dashboard go?** A new page "Risk" in `app.py`'s navigation. Its data module would follow
   `research_data` / `exposure_data` (`Reader.read("results", key, fn)`), and it would be tested with
   `AppTest` like the other pages.
7. **How are risk runs loaded today?** Only by `quant risk report RUN`, which renders from the `risk_*` tables.
   The dashboard would read those same tables.
8. **Valuation for stress.** Stress uses the snapshot exactly as the policy evaluation does.
   - Unvalued positions or a reconciliation gap above 1% make every stress scenario INCOMPLETE.
   - A stale position is shocked at its mark and flagged as conditional, because its price moves only when it
     trades again.
   - The suspension scenario uses the 60-session write-down convention (section 3, decision D3).

**Go / no-go for historical re-simulation: no-go.** Data and code identity are not recoverable, and 4 of the
6 configurations do not reconcile. No risk result will be attributed to a historical construction run.

## 3. Decisions to approve

| # | Decision | Recommendation | Reason |
|---|---|---|---|
| D1 | Construction integration | **A current re-simulation, never presented as the historical run.** `quant risk evaluate --construction C2 [--date D]` re-simulates the declared config now (research period, decision setting, both valuation curves), builds the snapshot at the date (default: the last session), and evaluates it. The target is `construction:C2@resim`. The run stores the config version, the data identity (current warehouse SHA, feature build) and a reconciliation table against the latest stored `ok` run of that config. The tolerance is a relative 1e-9 on every stored metric of the decision setting; it was declared on 2026-11-11 as the reproducibility policy (H-1), before this audit. The report headline says whether it reconciles. A non-reconciling re-simulation is still a valid risk evaluation of the declared rules on today's data, so the policy status is not changed by it. | Options A and B of the spec give nothing in practice: the grid is closed (one run per config, all failed), and new construction runs are not planned. Strict refusal would block 4 of 6 configs. |
| D2 | Where stress lives | Inside every risk evaluation (risk model `risk-v2`). Same snapshot and inputs, one run; `risk-v1` runs stay as stored, without stress. No separate `quant risk stress` command. `quant risk evaluate ... --no-stress` skips it. | One snapshot, so scenarios and limits can never describe different portfolios; less CLI surface. |
| D3 | Suspension scenario | The 3 largest positions by NAV weight, ties by symbol. Two durations: 20 sessions (exit blocked, still at the mark under the valuation rule; reports blocked weight and lost liquidity) and 61 sessions (past the 60-session write-down; reports the valuation loss). | Shows both the liquidity effect and the valuation effect without double counting: the write-down applies only at 61. |
| D4 | Market scenario | Two variants for each shock (-5%, -10%, -15%). **Beta-adjusted**: sum of w_i x beta_i x shock; INCOMPLETE if any position lacks a beta, as for the beta limit. **Uniform**: every position moves by the shock (invested share x shock), always computable, labelled "not beta-adjusted". Idiosyncratic return is 0 in both, and that is stated. | No false precision when betas are missing, while a computable reference remains. |
| D5 | Industry and cluster scenarios | -10% and -20% on each group, one at a time, on the positions of that group only. Unlabelled holdings are listed with their weight. A cluster scenario is INCOMPLETE while any holding lacks a cluster, as for the cluster limit. The report lists every group, sorted by loss. | Same conventions as the limits. |
| D6 | Liquidity scenario | ADV x 0.5 through `liquidity.position` (cost model v1 unchanged). Reports the liquidation cost, the maximum horizon and the illiquid weight, before and after; INCOMPLETE where ADV is invalid. | Reuses the PR 1 code; no formula is duplicated. |
| D7 | Scenario status | `MEASURED` or `INCOMPLETE` only. No threshold, no PASS / BREACH: scenarios are illustrations, not policy limits. Scenario losses are never summed. Valuation loss and liquidation cost are separate columns. | The spec's warning about implying an unapproved threshold. |
| D8 | Versioning and storage | Scenario set `stress/v1` (frozen, like policies), in a new table `risk_scenarios (set_id, version, config, created_at)`. Results go to `risk_stress_results` (run_id, scenario_id, family, group, status, nav_before, valuation_loss, valuation_loss_pct, cost, coverage, affected, reason, notes). The construction reconciliation goes to `risk_reconciliation` (run_id, metric, stored, resimulated, rel_diff, within). All additive. | Same pattern as `risk_policies`. |
| D9 | UI | A read-only page "Risk": choose a stored risk run (paper, backtest or construction); status and reasons, limits, Concentration, curves (pre-registered and traded marks shown apart), positions, stress tables, reconciliation, and the policy, scenario and model versions. It reads stored rows only and never recomputes. Optional: a "Risk: evaluate paper portfolio" task on the Run tasks page (it runs the CLI through the existing locked task runner). | The UI stays read-only on the databases; the existing task-runner pattern is reused. |

## 4. Implementation split (after approval)

1. `risk/stress.py` (pure scenarios) with hand-computed tests: market (both variants), industry, cluster
   (unclustered), liquidity (invalid ADV), suspension (tie-break, 20 versus 61 sessions, no double counting),
   sign conventions.
2. Persistence (`risk_scenarios`, `risk_stress_results`), `risk-v2` in `evaluate`, the report section, and
   `--no-stress`.
3. Construction: `portfolio` exposes a non-persisting `simulate_config(store, name, capital, costs,
   writedown_after)` that `portfolio.evaluate` itself then uses (one code path). Also a
   `risk.data.construction_snapshot`, the reconciliation table, and `--construction`. Tests: determinism, no
   change to `portfolio_results` or `research_runs` rows of the grid, a reconciliation pass and failure on a
   synthetic fixture, no hypothesis rows.
4. UI page and data module, `AppTest` tests (complete, INCOMPLETE, empty, failed run), read-only check.
5. Mutation check (the spec's list, `PYTHONDONTWRITEBYTECODE=1`), full suite, a real-store run (paper, one
   backtest, C1 to C6 construction), docs.

## 5. Acceptance (revised from the spec)

- Every scenario is deterministic, versioned (`stress/v1`) and labelled as an illustration. MEASURED or
  INCOMPLETE; no scenario is summed with another.
- Missing betas, labels, ADV or values give INCOMPLETE, never 0 or MEASURED.
- A construction risk run is labelled as a current re-simulation, carries its reconciliation table, and never
  writes to the grid's rows.
- The dashboard reads stored rows only; the UI tests cover complete, INCOMPLETE, empty and failed runs.
- Full suite green, CI synthetic; new tests mutation-checked.
- README, STATUS and this plan updated with the results.

## Result (2026-11-12)

Branch `risk-engine-pr2`. All decisions D1 to D9 were applied as approved. Full suite: 556 passed (534
before). CI stays synthetic: it checks code and fixtures, not the production store. The real-store results
below were run locally.

### Code

- `risk/stress.py`:
  - frozen scenario set `stress/v1`;
  - market (beta-adjusted, INCOMPLETE without a beta; uniform, labelled), industry and cluster (one group at a
    time; unlabelled holdings make the family INCOMPLETE), liquidity (ADV x 0.5 through `liquidity.position`),
    suspension (top 3, ties by symbol; 20 and 61 sessions);
  - results are MEASURED or INCOMPLETE only and never summed; untrusted positions make every scenario
    INCOMPLETE.
- `risk/store.py`:
  - all persistence, moved out of `evaluate.py`;
  - new tables `risk_scenarios` (frozen), `risk_stress_results` (with a `seq` column for the declared order)
    and `risk_reconciliation`.
- `evaluate.evaluate`: risk model `risk-v2`. Stress runs on the same assessed positions in the same run
  (`--no-stress` skips it). Re-simulation parameters and reconciliation rows are stored before the run is
  marked ok. A duplicate run id gives a clear error.
- `portfolio/evaluate.py`: `prepare` and `simulate`, shared by the grid and the Risk Engine; the grid's
  behaviour is unchanged (its tests pass).
- `risk/construction.py`:
  - re-simulates the configuration that the latest valid run stored (its version must match, or it is
    refused), under that run's stored universe, with both valuation curves;
  - builds positions from the trades at the engine's marks;
  - reconciles every stored strategy metric of the decision setting with a relative tolerance of 1e-9.
- CLI: `quant risk evaluate --construction C1..C6 [--date D]`, `--no-stress`. The report has new sections:
  "Re-simulation" and "Stress scenarios".
- UI:
  - a read-only "Risk" page: stored runs only, including failed ones; status, versions, limits, stress,
    positions, metrics, re-simulation;
  - a "Risk: evaluate paper portfolio" task on Run tasks.

### Tests and mutation check

New test files: `test_risk_stress.py` (13, hand-computed), `test_risk_construction.py` (7), and Risk page tests
(3) in `tests/ui/test_pages.py`.

Mutation check: 22 mutations for this PR, all caught; the 34 PR 1 mutations, rerun, are all caught.
- Two mutations first survived, and each got a test: the industry family using cluster labels, and the code
  constant `CONFIGS` used instead of the stored configuration.
- One mutation ("stored config ignored") is stopped by the version check in the code before any assertion
  runs. These are two independent defences, both tested.

### Real store

Backup `data/results.before-risk-pr2.duckdb`. After the runs, `portfolio_results` is identical. The only
changes are 3 new tables, 8 `risk` runs and their rows; there are no hypothesis rows.

- **Paper at 2026-11-09:** INCOMPLETE (cluster coverage; beta for CAR). Stress:
  - market uniform -5% / -10% / -15% of NAV; the beta-adjusted variant is INCOMPLETE (no beta for CAR);
  - worst industry (3510) -2.0% / -4.0%;
  - worst cluster (16) -2.9% / -5.9%, INCOMPLETE with 36.2% unclustered;
  - ADV halved: liquidation cost 10.7 -> 12.1 m VND, maximum horizon 0.55 -> 1.09 sessions, illiquid weight
    0% -> 14.6%;
  - suspension of the top 3: 17.3% of NAV blocked for 20 sessions, 17.3% lost at 61.
- **Backtest h10 k20 renew:** as in PR 1, plus stress.
- **Construction C1 to C6 (re-simulations at 2023-12-29):** all INCOMPLETE; none reconciles with its stored
  run.
  - C1 and C4: equity metrics are identical; only the counters and participation differ (C1: 6 metrics;
    C4: 4).
  - C2, C3, C5 and C6: most metrics differ, as the Phase 0 audit found.
  - Each report says so in its "Re-simulation" section.

## PR #9 review (external, 2026-11-12, head `700efb8`): assessment and plan

| Point | Verified | Assessment | Action |
|---|---|---|---|
| P1 the construction snapshot values positions apart from the engine's marks | Yes. The engine's mark is `last_price`: it is set to the buy price when the column has no mark yet, and updated at a traded close only while the position is held (`run_targets`, `_trade`). `_positions()` instead looks up the last traded close itself. The two differ when a buy session's close is not valid. Real runs: reconciliation gap 0.0, so no stored result is affected. | Valid. The valuation rule must not be duplicated. | `run_targets(..., snapshot_at=i)` records the engine's own state at the close of session i (shares, marks, cash, equity) in `PortfolioResult.state`. It is optional, and the grid's numbers are unchanged. `construction` uses that state. Regression test: a bought position whose close is not valid by the snapshot date is valued at the engine's buy-price mark, and positions + cash equal the engine's equity. |
| P1 reconciliation compares only the intersection of metrics | Yes. On C1, 5 of the 30 stored metrics were skipped silently: `break_even_extra_cost_per_side`, the three `excess_cagr_vs_*`, and `rank_persistence`. | Valid | Declared metric sets. **Required:** every strategy-simulation metric, plus `rank_persistence` and the three `excess_cagr_vs_*` (the benchmarks are recomputed with the grid's own functions). **Declared not recomputed:** `break_even_extra_cost_per_side` (a 16-simulation search), listed in the report with that reason. A required metric missing on either side, an empty comparison, or a NULL is not a match. The only exception is `participation_*`, where NULL on both sides means "no order with a valid ADV", and that is stated. A stored metric outside both sets is listed as "not compared". "Reconciles" requires every required metric compared and within tolerance. A nullable `note` column is added to `risk_reconciliation` (additive `ALTER ... IF NOT EXISTS`). |
| P1 stress accepts non-finite inputs | Yes. A beta of +/-inf is not NaN, so it passes `isna()`. The liquidity shock relies on upstream normalization. | Valid | Validation at the stress boundary with `np.isfinite`: a non-finite or missing beta makes the beta-adjusted scenario INCOMPLETE. ADV is validated locally with `liquidity.valid_adv`. A non-finite position value or weight, or NAV, makes every scenario INCOMPLETE. Direct tests with NaN, +inf and -inf, without upstream normalization. |
| P2 suspension wording | Partly. The note says "valued at 0 (write-down after 60)". | Valid | Wording "assumed write-down under prolonged suspension (valuation rule), not an expected loss or a forecast" in the note, the report text and the UI caption. |
| P2 open-price fallback test | Same as P1 | - | The regression test above. |
| P2 liquidity input contract | Same as P1 | - | Local validation (above). |

Mutation check of each new test (`PYTHONDONTWRITEBYTECODE=1`). The real construction runs are re-run.

### PR #9 review fixes: result (2026-11-12)

Full suite: 565 passed (556 before).

- **Engine state.** `run_targets(..., snapshot_at=i)` keeps the engine's own state at that close in
  `PortfolioResult.state`: shares, marks, last trade, cash, equity. The construction snapshot reads it, so no
  valuation rule is re-derived. Default `None`: the grid's numbers are unchanged (backtest and portfolio tests
  pass, `portfolio_results` identical).
  - Regression test: a bought position with no valid close up to the snapshot is valued at the engine's
    buy-price mark, and positions + cash equal the engine's equity.
- **Reconciliation contract.**
  - Required: the 25 simulation metrics, plus `rank_persistence` and the three `excess_cagr_vs_*`, recomputed
    with the grid's own functions.
  - Declared not recomputed: `break_even_extra_cost_per_side`, with its reason.
  - A missing required metric, NULL (except `participation_*` NULL on both sides, as declared) or an empty
    comparison is not a match. Any other stored metric is listed as not compared.
  - "Reconciles" requires all 29 required metrics compared and within 1e-9.
  - `risk_reconciliation.note` was added (`ALTER ... ADD COLUMN IF NOT EXISTS`).
  - Found while fixing: the comparison returned `numpy.bool_`, which an identity check (`is True`) rejects;
    values are now cast to Python types.
- **Stress boundary.**
  - A beta that is NaN or +/-inf makes the beta-adjusted scenario INCOMPLETE.
  - ADV is validated in the liquidity shock itself, through `liquidity.valid_adv`.
  - A non-finite position value or weight makes every scenario INCOMPLETE.
  - Direct tests with NaN, +inf and -inf, without upstream normalization.
- **Wording.** The 61-session suspension is described as an "assumed write-down under prolonged suspension ...
  not an expected loss or a forecast" in the note, the report and the UI.
- **Mutation check.** 30 mutations for PR 2 (22 updated plus 9 new; two re-anchored), all caught; the 34 PR 1
  mutations are all caught. Two are caught by an error rather than an assertion, and in both cases the code
  never yields a false measurement:
  - removing the local ADV check makes the liquidity model return no cost, so the sum raises;
  - the "stored config ignored" mutation is stopped by the version check.
- **Real store, construction re-runs.** All six still do not reconcile. With the full required set, C1 matches
  23 of 29 and C4 matches 24 of 29; C4's `rank_persistence` also differs. C2, C3, C5 and C6 differ on 20 to 22
  metrics. The NAV reconciliation gap is 0.0 everywhere, and `portfolio_results` is identical to the backup.

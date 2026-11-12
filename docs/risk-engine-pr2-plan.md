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

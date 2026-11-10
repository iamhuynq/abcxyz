# Plan: fixes from the repository review of 2026-11-10

<!-- type: refactor -->
<!-- status: proposed 2026-11-10 -->

Source: `vn-stock-quant-main-review-2026-11-10.md` (review of `main` at `ba9d6e5`), and the reviewer's
follow-up after our rebuttal. Every finding was checked against the code and the real data first. The
reviewer accepted downgrading its P1 to a design issue (P2) and added two requests: split the
sensitivity check by mask, and keep the full random-control distribution. No validation, holdout or
forward data is used for a decision.

## Findings checked

| # | Finding | Code | Measured on real data |
|---|---------|------|-----------------------|
| 1 | The research universe mixes point-in-time filters with outcome masks | True: `ResearchParams.universe_sql()` adds `fwd_has_price_jump_20d`, `entry_blocked` and `crosses_period` to `is_traded`, ADV, age, `price_jump` and `bad_source_date` | Research period, liquid universe (959,313 rows): future jump 0.02%, entry blocked 0.41%, crosses period 0.69%. Blocked rows average +1.23% 10-session excess against +0.09% for kept rows |
| 1b | Masks ignore the horizon (found while checking 1) | True: `crosses_period` and `fwd_has_price_jump_20d` use the 20-session window for the 5- and 10-session targets too | Not measured yet; over-censors short horizons near period ends |
| 2 | `daily.sh` exits 0 when another run holds the lock | True: `echo "Another run (pid ...) is in progress."; exit 0`. The UI starts it with `own_lock`, so a refused run shows `ok` | Not hit so far |
| 3 | Raw files can be overwritten within one second | True: `RawStore.write` names files `{chunk}__{HHMMSS}.json.gz` and uses `os.replace` | Not hit so far; breaks the "raw is never modified" promise |
| 4 | 20 random controls give a noisy "share beaten" | True: `N_CONTROL = 20` in `econ.py` and `portfolio/evaluate.py`; `N_RANDOM = 20` in `backtest/runner.py` and `daily.py` | One run moves the share by 5 points |

Affected by 1: pattern engine (`engine.py`), decile scan, interaction scan, event study, cross group tests,
cross hypotheses, the daily report and the q-value baseline (`results.py`). Not affected: backtests, the
paper portfolio, economic evaluation and portfolio construction (`backtest/data.py` does not use these
filters).

## Fixes

**1. Point-in-time universe and separate outcome masks.**
- Split `universe_sql()` into:
  - `universe_sql()`: point-in-time only (`is_traded`, ADV, age, `NOT price_jump`, `NOT bad_source_date`);
  - `mask_sql(masks, horizon)`: the outcome masks, each named:
    - `quality`: no data-error price jump inside the target window;
    - `execution`: the next open is not limit-up (`entry_blocked`);
    - `period`: the target window does not cross a period boundary.
  - `sample_sql(horizon)` = universe AND the masks chosen in `ResearchParams`.
- Per-horizon masks (fix 1b): add `jump_ahead_{h}d` and `crosses_period_{h}d` for h = 5, 10, 20 to
  `04_stock_targets.sql`. The existing 20-session columns stay unchanged.
- **Recorded definitions do not change.** `ResearchParams` gets
  `outcome_masks = ("quality", "execution", "period")` and `mask_window = "legacy20"` as defaults. These
  give exactly today's SQL: patterns, the scan, the interaction scan and validated or frozen definitions
  are untouched.
  - A test asserts that the default `sample_sql` equals today's `universe_sql()` text.
  - New research definitions use `mask_window = "horizon"`.
  - The parameter JSON stored with each run changes only when a run uses non-default values (checked
    against the duplicate-run guards and the registry hash check before the change).
- Call sites move from `universe_sql()` to `sample_sql(horizon)`, where the horizon is known (the target
  column).

**2. Mask sensitivity report** (`quant audit masks`, research period only, read-only on decisions).
- Five configurations:
  - baseline: universe only;
  - quality only;
  - execution only;
  - period only;
  - combined: today's definition.

  Each runs with the legacy 20-session window. Combined also runs with the per-horizon window.
- What is compared for each configuration:
  - eligible rows, per year and per industry;
  - the distribution of the 5/10/20-session excess (mean, quantiles);
  - for every recorded research-period pattern and scan feature: rows kept, lift, t and p, whether it
    passes its threshold, and its rank;
  - for the interaction scan: the 54 contrasts, the candidate rule, and Spearman rank agreement;
  - the share of each mask's removed rows that falls inside each pattern's hits (concentration). The
    reviewer is right that a small overall share can still sit inside one pattern.
- Sensitivity runs are a new run kind `mask_audit`. They log **no** hypotheses, so the multiple-testing
  families do not grow and no q-value changes.
- Decision rule, declared before running:
  - A recorded research-period conclusion is "mask-sensitive" if, under the baseline, its pass/fail flips
    or its effect changes sign.
  - A mask-sensitive conclusion is marked in the report.
  - A registry state changes only through `quant registry move` with a reason quoting the report, and
    only for research-stage hypotheses.
  - Validated, failed or forward hypotheses (for example the avoid signal, PASS) are **not** re-decided:
    validation and holdout cannot be reused. Their robustness is reported descriptively.
  - Old runs are never deleted. A run is invalidated only if its own definition is shown to be wrong; a
    different but valid mask choice is not a reason to invalidate.
- Output: `docs/mask-sensitivity-results-2026-11-XX.md`.

**3. `daily.sh` "already running" exit code.**
- Exit **9** with the message "Another run (pid N) is in progress; nothing done."
- `EXIT_MEANINGS[9]` = "busy: another run holds the pipeline lock". `TaskRecord.status()` reports exit 9
  as `refused`, not `failed` or `ok`.
- "Already done for DATE." stays exit 0 (correct: nothing to do).
- Tests:
  - the lock lint test runs `daily.sh` against a held lock and expects exit 9;
  - the UI record shows `refused`.

**4. Collision-proof raw files.**
- Name: `{chunk}__{HHMMSS}_{microseconds}_{4 random hex}.json.gz`. Readers sort on the part after `__`, so
  `iter_files` ordering stays chronological and old names still parse (checked by a test with mixed
  names).
- Create the target with `os.link(tmp, target)` and then remove `tmp`. Linking fails if the target
  exists, so no file is ever replaced. A collision retries with a new suffix; a second collision raises.
- Tests:
  - two writes with the same `fetched_at` give two files with both payloads;
  - an existing target is never overwritten;
  - the old-name ordering still works.

**5. Random controls: 200 runs and the full distribution.**
- `N_CONTROL` / `N_RANDOM` = 200 for **new** economic, portfolio and backtest evaluations. Each run
  stores:
  - every control's CAGR (a new table `control_outcomes`);
  - the 5/50/95 percentiles;
  - the empirical exceedance `(1 + #controls >= strategy) / (1 + n)`.
- Pass criteria keep their meaning as a share: 95% beaten.
- **Pre-registered tests keep 20.** The backtest pre-registration fixes "20 runs, seeds 0-19, beat at
  least 19 of 20", and the paper portfolio follows it. Their verdict stays on seeds 0-19.
  - The paper portfolio adds 200 controls as a separate diagnostic line in the report and the UI, labelled
    "not the pre-registered criterion".
  - Recorded econ and portfolio runs are not re-run: their verdicts would not change, because no
    configuration passes even before the controls.
- Runtime: about 10x for the control part. This will be measured; if `quant econ evaluate` exceeds about
  15 minutes, controls run in a process pool.

## Not in scope (backlog)

- `Retry-After` given as an HTTP date.
- A Benjamini-Yekutieli sensitivity column next to BH q-values.
- An `adj_ratio` audit across the whole overlap of adjusted and raw history.
- A backup and restore runbook.
- A vendor-schema canary: compare `fireant probe` against a stored snapshot.

## Order and checks

1. Fix 3, then fix 4: small and independent; one commit each.
2. Fix 1: the default SQL is identical to today's, checked by a test. Then rebuild `research.duckdb` for
   the per-horizon columns.
3. Fix 2: run the audit and write the results doc.
4. Fix 5.
5. Every new test is mutation-checked. Run the full suite and the GNU date/stat shim test. Update
   `STATUS.md`, the data and feature dictionaries, and this plan's Result section.

Branch `review-2026-11-10-fixes`, one commit per fix; the user pushes and merges.

## Rollback

- Code: revert the branch; `main` is unchanged until merged.
- `research.duckdb` is rebuilt atomically from the warehouse; rebuild from `main` to roll back.
- Before the audit and the fix-5 runs, back up `results.duckdb` to `data/results.before-review-2026-11-10.duckdb`.
  The audit adds runs and logs no hypotheses; restoring the backup undoes it.
- Raw store: the new names are additive; old files are untouched.

# FireAnt contract check and BY q-values (plan, 2026-11-11)

Source: the post-merge checklist for PR #4 (external, 2026-11-11), section 5, priority P2. The user picked
two items: A (FireAnt contract validation) and B (multiple testing under dependence). Both are already in the
`STATUS.md` backlog ("a vendor-schema canary", "a BY column next to BH").

## What the checklist asked, and what is already done

Sections 1 to 4 and 6 of the checklist were checked on 2026-11-11 and need no new work:

- `main` = `origin/main` = `730464a`, clean; `uv run pytest -q`: 406 passed.
- Raw store: a file is written to a `.tmp` name and then `os.link`-ed to its final name. A file that was
  only partly written never gets a final name, and readers glob `*.json.gz` only.
- Controls: the denominators of `n_controls` and the CAGR percentiles are documented in `controls.py`.
- Mask audit: its report records commit, warehouse hash, feature build and code hash, and separates
  statistical sensitivity from decision sensitivity. It reads research data (<= 2023), so a re-run would
  give the same result.
- Run records: run IDs, parameters, provenance and invalidation are already in `results.duckdb`.

## A. FireAnt contract check

### Problem

The normalizers read fields with `raw.get(key)`. If FireAnt renames or drops a field:

- quotes: an unknown key already raises (`normalize_quotes`), but a missing key becomes NULL silently;
- corporate actions, report marks, symbols, fundamental, ICB: unknown keys are ignored, and missing keys
  become NULL silently;
- a type change (for example a number that becomes a string, or a date in another format) is caught only
  when a parser fails, or not at all.

Today the warehouse checks in `fireant validate` notice some of this later, as NULL prices. They do not
notice it for the less checked fields (foreign flows, corporate action dates, report marks).

### Design

1. `src/fireant_crawler/validate/contract.py`:
   - one contract per job (`quotes`, `corporate_actions`, `report_marks`, `fundamental`, `icb`,
     `universe`);
   - for each field it gives the allowed JSON kinds (`number`, `string`, `bool`, `null`), whether the field
     is required (the key must be present), and for date fields the expected format;
   - the field lists are written from the shapes observed in `data/raw` and from the keys each normalizer
     reads.
2. `check_payload(job, data) -> list[Violation]`, where a violation has a level:
   - **error**: a required key is missing, a value has a kind the contract does not allow, a date does not
     parse, or the payload is not the expected list or object;
   - **warning**: an unknown key (except for quotes, whose normalizer already rejects unknown keys).
3. `fireant validate` also checks the raw files of the latest fetch date of each job. Contract errors
   count as error-level findings, so `daily.sh` stops before the build (existing exit 8). The validation
   report gets a "Contract" section that lists each violation once, with its count and an example of the
   file and key. It never shows values.
4. `fireant contract scan [--job J] [--since YYYY-MM-DD]`: checks every stored raw file (read-only). This is
   the check against real data. CI cannot do it, because `data/` is never committed.
5. Tests:
   - each normalizer reads only keys that its contract declares (from `FIELD_MAP`, `_FUNDAMENTAL_MAP` and
     the explicit keys), so the contract and the code cannot drift apart;
   - violations, one test per kind: a missing key, a wrong kind, a bad date, an unknown key, a wrong
     payload shape;
   - the validate path: a contract error makes `fireant validate` exit non-zero, and a warning does not.

### Verification on real data

- Run `fireant contract scan` over the whole `data/raw` (about 26,000 files) before merging.
- Expected result: zero errors. Anything found is either a real past change, which gets recorded and
  explained, or a contract that is too strict, which gets fixed. No contract is loosened without writing
  down why.

## B. BY q-values next to BH

### Problem

`hypothesis_q` adjusts with Benjamini-Hochberg over every logged test of every valid run. BH controls the
false discovery rate under independence or positive dependence. Our tests overlap heavily (the same stocks,
dates and horizons), and positive dependence is plausible but not proven. Benjamini-Yekutieli is valid under
any dependence, at the cost of a factor c(m) = 1 + 1/2 + ... + 1/m (about 9 for m = 5,000).

### Design

1. `stats.benjamini_yekutieli(p_values)`: BY q-values in the input order, capped at 1.
2. The `hypothesis_q` view gets a `q_by` column, computed by the same ranking:
   `least(1, min(p * m * c(m) / rank) over ranks >= rank)`. `q_value` and every recorded decision stay as
   they are. The view is recomputed on every read, so no stored row changes.
3. `quant audit fdr` writes a descriptive report, `data/reports/research/audit-fdr-<ts>.md`, with:
   - m, c(m) and the provenance header used by the other audits;
   - every logged test with BH q < 0.05, its BY q, and whether it stays under 0.05;
   - for every registry hypothesis, its linked runs (the same link that `quant registry show` uses), with the
     minimum BH q and the minimum BY q of their tests, next to the recorded decision.
   The report changes no registry state and logs nothing.
4. Tests:
   - BY matches a hand-computed example;
   - BY >= BH element-wise;
   - the SQL `q_by` equals the Python function on the same p-values;
   - the view still returns the old `q_value` unchanged.

### Interpretation rule (declared before running)

BY is a robustness diagnostic. A finding that passes BH but not BY is reported as "not robust to arbitrary
dependence". Its registry state does not change. Any state change would go through `quant registry move`,
with the user's decision, as usual.

## Out of scope

- Cost model, historical exchange and limit bands (P2, group D): not done. They need a source of historical
  bands that we do not have yet.
- Retry-After as an HTTP date, the `adj_ratio` overlap audit, and backup and restore (P3, group C): next
  milestone.
- No change to recorded definitions, pre-registrations, `registry_seed.py` or stored runs.

## Steps

1. Branch `contract-and-fdr` from `main`.
2. A: contract module, checks, `fireant validate` integration, `fireant contract scan`, tests.
3. A: scan all of `data/raw`, and record the result in this doc.
4. B: `benjamini_yekutieli`, `q_by` in the view, `quant audit fdr`, tests.
5. B: run `quant audit fdr` on the real store, and record the result in this doc.
6. Mutation-check every new test. Run the full suite and the GNU date/stat shim test.
7. Update `README.md` (new commands), `STATUS.md`, and add a Result section to this doc.
8. Commit on the branch. The user pushes and opens the PR.

## Rollback

- Code only, on a branch. Reverting the merge commit removes it.
- The view change is `CREATE OR REPLACE VIEW` on each open. After a revert, the old definition is recreated
  on the next open. No stored data changes.
- If the contract check wrongly stops `daily.sh`, either revert, or run `fireant validate --no-contract`
  (a flag added in step 2) for that day, and fix the contract.

## Success criteria

- `fireant contract scan` over all of `data/raw`: zero errors, or every error explained in this doc.
- A synthetic renamed field in a quote, corporate action or report mark payload stops `fireant validate`
  (exit non-zero).
- `hypothesis_q.q_value` is the same as before for every row; `q_by >= q_value`.
- The `quant audit fdr` report exists, and the registry is unchanged (same rows, same states).
- Full suite green; every new test fails under a mutation of the code it covers.

## Result (2026-11-11)

Branch `contract-and-fdr`. Full suite: 435 passed (406 before). `daily.sh` is unchanged, so the GNU date/stat
shim run was not repeated; the static check for BSD-only flags is part of the suite. Every new test was
mutation-checked: 20 mutations, all caught. Two of them first survived and got targeted tests: the SQL `q_by`
without c(m), and the link through transitions.

### A. Contract check

- `src/fireant_crawler/validate/contract.py`: seven contracts (quotes, corporate actions, report marks,
  fundamental, ICB, universe search, universe symbol). `/instruments` is raw-only and is skipped.
  - Fields are the ones seen in the whole raw store. NULL is allowed only where it was seen: the three
    proprietary-trading values in quotes, `executionDate`, fundamental values, and symbol exchange / ICB /
    industry codes.
  - Integer fields (`eventID`, event `type`, `companyType`, ICB `level`) reject floats.
- `fireant validate` checks the raw files of each job's latest fetch date (5,583 files; the whole command takes about 6 s). A
  contract error makes the command exit 1, so `daily.sh` stops before the build (exit 8, as before). The
  validation report has a new "Contract" section. `--no-contract` skips the check.
- `fireant contract scan`: all 26,428 stored raw files, **0 errors, 0 warnings**; one `/instruments` file is
  skipped. The 2026-11-11 survey of about 6.6 million rows found every field present in 100% of rows, and
  every date in ISO format.
- Test: every key that each normalizer reads is declared in its contract. A recording dict runs the real
  normalizers on the fixtures.

### B. BY q-values

- `stats.benjamini_yekutieli`, which matches `statsmodels` `fdr_by`. The `hypothesis_q` view has a new
  `q_by` column, and `q_value` is unchanged (the existing BH test passes).
- `quant audit fdr`: `data/reports/research/audit-fdr-20261111T093526.md`, on the real store.
  - m = 295 logged tests, c(m) = 6.266. 120 tests pass BH at 0.05; 86 of them also pass BY.
  - Section 2 uses each hypothesis's own test. For the interaction hypotheses, that is their own
    `ix_<factor>_<dimension>` test of the shared scan. A first draft took the minimum over the whole scan
    run, which would have shown every interaction at the scan's best q. This was caught on the real report
    and fixed, with a test.
- Registry hypotheses that pass BH but not BY (descriptive; no state changes):

| hypothesis | state | q BH | q BY |
|---|---|---|---|
| P7-IX-foreign_flow-breadth | candidate | 0.0342 | 0.215 |
| P7-IX-liquidity-liquidity | candidate | 0.0382 | 0.239 |
| P7-IX-order_flow-foreign | candidate | 0.0351 | 0.220 |
| P7-IX-reversal_1w-liquidity | candidate | 0.0270 | 0.169 |
| P4b-G2 (industry momentum) | failed | 0.0213 | 0.133 |
| P6-B1 (portfolio) | failed | 0.0210 | 0.132 |

- Interpretation, as declared in this plan:
  - Four of the 13 open interaction candidates are not robust to arbitrary dependence. The other nine
    pass BY. `P7-IX-reversal_1w-foreign` is borderline (BY q 0.0498).
  - P3-C3 (validated avoid signal) and P4b-G3 pass BY by a wide margin.
  - P4b-G2 and P6-B1 had already failed on other grounds.
  - Nothing changes in the registry. Whether to note BY in a future pre-registration of the interaction
    themes is the user's decision.

### CI fix: lock released before the run record ends

The first CI run of the branch failed `tests/ui/test_tasks.py::test_takes_over_a_stale_lock`. The code at
fault was already on `main`, outside this branch's changes.
- The cause: `stock_ui/task_runner.py` saved the final run record (`ok`) first, and released the pipeline
  lock afterwards, in `finally`. A reader could see `ok` while the lock still existed.
- The slow CI runner hit that window. Reproduced locally by delaying `locks.release` by 1.5 s: 3 of the 13
  task tests failed.
- Fix: release the lock first, then save the final record. Until that save, the record still shows the run
  as running, so no other task can start in between.
- With the same delay, all 13 tests pass. Full suite: 435 passed.

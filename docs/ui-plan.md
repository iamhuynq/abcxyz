# Plan: Management UI (Streamlit, local only)

<!-- type: feature -->
<!-- status: built 2026-10-05 (phase 1: A, B, C; phase 2: D, E, F) -->

Decisions taken with the user (2026-10-05): screens A-F, **A + B + C first**; runs **only on this Mac**;
**Streamlit**; daily pipeline first, then UI (done: `docs/daily-pipeline-plan.md`).

## Goals

| Screen | Purpose | Phase |
|--------|---------|-------|
| A. Overview | Is the system healthy? Data freshness, last pipeline run, failed tasks, schema, token expiry, disk | 1 |
| B. Run tasks | Start the existing commands with a button, follow the live log, cancel, see history | 1 |
| C. Symbol lookup | One stock: adjusted price chart, volume, foreign flow, order imbalance, corporate-action and report markers, latest features, events it triggered | 1 |
| D. Research | Runs, hypothesis log with q-values, invalidate a run (with reason) | 2 |
| E. Backtest + paper | Backtest runs and equity curves, forward paper portfolio vs random band | 2 |
| F. Daily report | Daily reports by date, avoid list, forward scoreboard | 2 |

Non-goals: no new research logic, no editing of data, no remote access, no order placement.

## Key design decisions

| Topic | Decision | Why |
|-------|----------|-----|
| Access | `127.0.0.1` only, headless, usage stats off (`.streamlit/config.toml`) | Local only; the token on this machine can place orders |
| Read path | The UI **never writes** a database. Each query opens a `read_only=True` DuckDB connection and closes it right away; results cached (`st.cache_data`, TTL 60 s) | DuckDB allows one writer **or** many readers per file across processes. A UI holding a connection open would make the 18:30 `fireant update` fail |
| While a writer runs | If `data/.daily.lock` exists, or opening fails with a lock error, show the **last cached result** with a banner "Updating - showing data as of HH:MM" instead of an error | Requested behaviour; no lock fight with the pipeline |
| Race guard on the writer side | Writers (`Warehouse` read-write, `ResultsStore`) retry opening for up to 60 s on a lock error | Covers the sub-second window where a UI read is in flight when the pipeline starts |
| How tasks run | Buttons start the **existing CLI commands** from a fixed allowlist (no free-text command). A small runner (`python -m stock_ui.task_runner <task>`) is launched detached (own process group), takes the **same lock** as `scripts/daily.sh`, writes the log to `data/logs/ui/` and a record to `data/logs/ui/tasks.jsonl` | One code path for CLI, launchd and UI; a task survives closing the browser; UI tasks and the scheduled run can never overlap |
| Task list (phase 1) | `daily.sh` (full pipeline), `fireant update`, weekly `update --jobs report_marks,fundamental`, `fireant validate`, `fireant status`, `fireant migrate --dry-run`, `quant build`, `quant daily`, `quant daily --date D` | Long or risky commands (`backfill`, `migrate` apply, `normalize`, holdout runs) stay CLI-only |
| Cancel | SIGTERM to the task's process group | The crawler commits per task, `quant build` replaces the file atomically, `quant daily` re-runs cleanly; all are safe to interrupt |
| Exit codes | Shown with meaning: 0 ok, 3 token expired, 4 schema out of date, 5 holdout locked, 7 session incomplete, 8 validation errors, 9 busy (shown as refused), 10 offline | Same codes as the CLI and `daily.sh` |
| Token | Shown only as expiry date, days left and the orders-write warning (`inspect_token`); the token string never reaches the page | Same rule as the CLI |
| Charts | Plotly (candlestick + subplots) | Streamlit's built-in charts cannot draw candlesticks with markers |
| Dependencies | `streamlit`, `plotly` in a separate uv group `ui` | The crawler and research code do not need them |
| SQL | Symbol and dates are bound parameters only | Parameterized-queries rule |

## Layout

```
src/stock_ui/
  app.py              # st.navigation, sidebar (lock banner, data date)
  db.py               # read-only query helper, lock detection, last-good cache
  status.py           # facts for screen A (pure functions over read-only connections)
  tasks.py            # allowlist, start/cancel, history, exit-code meanings
  task_runner.py      # detached runner: lock -> run -> record
  symbol_data.py      # queries for screen C
  charts.py           # plotly figures
  pages/overview.py, pages/tasks.py, pages/symbol.py
scripts/ui.sh         # uv run --group ui streamlit run src/stock_ui/app.py --server.address 127.0.0.1
.streamlit/config.toml
```
Each file stays under 400 lines.

## Screen contents (phase 1)

**A. Overview**
- Pipeline: last successful session (`data/.daily_done`), lock held or not (PID, age), last `daily-*.log`
  result line, launchd installed or not (plist present), next expected run.
- Data: latest date in the warehouse, rows on that date vs the previous session (same coverage rule as
  `quant daily`), last API call time, failed crawl tasks (count + 10 latest with error).
- Research DB: feature set version, build time, whether it matches the current warehouse hash.
- Daily scan: latest `daily_scans` rows (status, coverage); link to the latest daily report.
- Validation: date of the latest validation report and its error/warn counts.
- Schema: up to date or N pending changes (read-only check, same as `fireant status`).
- Token: expiry and days left (warning under 14 days), orders-write warning.
- Disk: size of raw/, warehouse, research, results.

**B. Run tasks**
- One button per allowlisted task, disabled while any task or the scheduled run holds the lock (shows who).
- Running task: live log tail (refresh every 2 s), elapsed time, Cancel.
- History: last 50 runs (task, start, duration, exit code + meaning, log link).
- After a task finishes, the read caches are cleared so A and C show the new data.

**C. Symbol lookup**
- Search box over listed and delisted symbols (code, name, exchange, industry).
- Header: name, exchange, ICB industry, trading span, listed/delisted, fund flag.
- Chart (date-range selector, adjusted by default, raw toggle): candlestick + MA20/50/200; volume;
  foreign net value; order imbalance. Markers: corporate-action ex-dates (type in the tooltip),
  report dates (BCTC), data-quality flags (`price_jump`, `foreign_inconsistent`).
- Latest features table (from `rs.stock_features`) and the symbol's `daily_events` history.

## Tests

- `db`: a read while another **process** holds a read-write connection returns the last cached result and
  the "updating" flag; no exception reaches the page.
- Writer retry: a read-write open succeeds once a short-lived reader in another process closes.
- `task_runner`: refuses while the `daily.sh` lock exists; records exit codes; cancel stops the process
  group; a stale lock (dead PID) is taken over, same rule as `daily.sh` (stub commands, as in
  `test_daily_script`).
- Allowlist: unknown task names are rejected; no shell string is built from user input.
- Page smoke tests with `streamlit.testing.AppTest` on the synthetic warehouse: A, B, C render without
  errors; database file modification times unchanged afterwards (read-only proven); the token value
  never appears in the rendered output.
- Full suite stays green (currently 156).

## Pre-mortem

- **Most likely failure**: the UI holds a DuckDB lock at 18:30 and `fireant update` fails. Mitigations:
  connections are opened per query and closed; the UI does not read while the pipeline lock exists;
  writers retry for 60 s. Test above reproduces the cross-process case.
- **Second**: a task started from the UI overlaps a launchd run. Mitigation: the same mkdir lock.
- **Third**: a heavy chart query on the 1.8 GB research DB is slow. Mitigation: per-symbol queries only
  (filtered by symbol, bound parameter), cached.

## Rollback

- Additive: delete `src/stock_ui/`, `scripts/ui.sh`, `.streamlit/`, the `ui` dependency group
  (`uv remove --group ui streamlit plotly`) and `data/logs/ui/`. No schema or data changes.
- The writer retry is a small change in `Warehouse` / `ResultsStore`; revert it on its own if needed.

## Open questions (defaults used unless you say otherwise)

1. UI language: **English** labels (same as reports and code), or Vietnamese?
2. Port: **8501** (Streamlit default).

## Result (2026-10-05, phase 1)

Run: `scripts/ui.sh`, then open http://127.0.0.1:8501 (stop with Ctrl+C). Verified listening on
127.0.0.1 only (the LAN address is refused).

- `src/stock_ui/`: `db.py` (read-only Reader), `locks.py` (lock shared with daily.sh), `tasks.py` +
  `task_runner.py`, `status.py`, `symbol_data.py`, `charts.py`, `app.py`, `pages/{overview,tasks,symbol}.py`.
- `fireant_crawler/store/locking.py`: `connect_with_retry` (60 s) used by `Warehouse` and `ResultsStore`.
- The `ui` dependency group (streamlit 1.65, plotly 7.1) is a default uv group, so tests can import it.
- 180 tests pass (24 new in `tests/ui/`). The cross-process lock cases use a real second process.
- Checked on real data: all three pages render, every `*.duckdb` file is unchanged afterwards, and the token
  never appears in the rendered output.

Changes after code review:
- The Reader now ignores a stale pipeline lock (dead pid or > 3 h). Before, a lock left behind by a crash
  blanked every page.
- Exited runners are reaped (`waitpid`). Before, a runner killed hard became a zombie, showed "running"
  forever and kept every button disabled.
- A lock directory without a pid file that is under 10 s old counts as held, because daily.sh is still
  creating it. A stale lock is moved away with an atomic rename before it is retaken.
- Run ids carry a random suffix and records are created exclusively. Cancel tolerates a run that already
  ended.
- The cache is invalidated when a database file changes (inode, mtime or size of the file and its WAL), so
  every tab sees new data right after a task. The task buttons redraw whenever the busy state changes.
- The runner pins `DATA_DIR` for the command, so the command and the UI always use the same lock directory.
- Symbol page: the selectbox is keyed by symbol, not by position. The raw-price view never silently shows
  adjusted prices. Markers are guarded against empty price data.

Known limit: if the runner itself is killed with SIGKILL, its command keeps running orphaned while the
lock looks stale. DuckDB's own file lock still prevents two writers on the same file; the second one waits
60 s and then fails.

## Result (2026-10-05, phase 2)

- **D. Research** (`pages/research.py`):
  - Runs filtered by kind, period and status, each with its 10-session lift vs the universe and its q-value.
  - Run detail: definition SQL, lifts, comparison with the base pattern, segment statistics, scan IC, logged
    tests.
  - Hypothesis log: the whole BH family (m), with a q < 0.05 filter.
  - Validation decisions, shown verbatim.
- **Invalidate a run**: a form with a required reason and a confirmation checkbox. It starts the hidden task
  `invalidate_run` (`quant runs --invalidate=RUN --reason=TEXT`) through the task runner under the shared
  lock, so the UI itself still never writes. The values use the `=` form so a reason may start with `-`.
  `quant runs --invalidate` now exits 1 when no run matches; before, it reported success.
- **E. Backtest and paper** (`pages/backtest.py`):
  - The forward paper portfolio against the 5-95% band of random controls, equal weight and VNINDEX, plus
    open positions. It shows "not started" until two forward sessions exist.
  - Backtest runs with their key metrics. Selecting one shows the equity curve against random median, equal
    weight and VNINDEX, the drawdown, a metrics pivot, returns by year, exit reasons, trades and parameters.
- **F. Daily report** (`pages/daily.py`): the rendered report for any date, a filterable event table per
  session with the avoid list highlighted, and the scan status.
- 189 tests pass (33 in `tests/ui/`). They include an end-to-end invalidate through the real CLI and page
  tests on synthetic data with forward sessions, which covers the paper portfolio.
- Checked on real data: all six pages render, the `*.duckdb` files are unchanged afterwards, and the token is
  never shown. A page crashed when no backtest had ever run (the table was missing); it now shows an info
  message instead.

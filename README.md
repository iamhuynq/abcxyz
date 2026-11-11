# Vietnam stock quant research system

A personal research system for the Vietnamese stock market. It covers the FireAnt data crawler,
point-in-time features, pattern and event research with pre-registered validation, cross-stock and industry
analysis, a portfolio backtest, a daily pipeline with forward paper trading, and a local Streamlit UI.

It is not investment advice. No signal tested so far is profitable after trading costs; see `docs/`.

## Setup

```bash
uv sync                      # Python 3.12, dependencies from uv.lock
cp .env.example .env         # then paste your FireAnt token into .env (never commit .env)
```

Data lives in `data/` (git-ignored). The full history is built with `uv run fireant backfill` (hours, rate
limited to 2.5 req/s). After that:

```bash
scripts/daily.sh             # daily pipeline: update -> validate -> build -> scan, paper trading, report
scripts/ui.sh                # local UI on http://127.0.0.1:8501 (Ctrl+C to stop)
```

## Commands

| Command | Purpose |
|---------|---------|
| `uv run fireant update` / `backfill` / `status` / `validate` | Crawler (GET-only, whitelisted paths) and data checks; `validate` also checks the latest raw payloads against their contracts (`--no-contract` skips that) |
| `uv run fireant contract scan [--job J] [--since DATE]` | Check stored raw files against the payload contracts (read-only) |
| `uv run quant build` | Rebuild `research.duckdb` (features, targets, event catalog) from the warehouse |
| `uv run quant daily` | Scan new sessions, paper portfolio, daily reports |
| `uv run quant pattern` / `scan` / `backtest` | Research runs (research period by default; other periods need `--prereg`) |
| `uv run quant cross build` / `describe` / `test` | Cross-stock and industry analysis |
| `uv run quant events study` | Descriptive event study |
| `uv run quant factors describe` | Factor structure report (Factor Engine f1; descriptive, no returns) |
| `uv run quant interactions scan` | Factor IC by market regime (research period only, logged, one run) |
| `uv run quant econ evaluate` | Economic evaluation: cost model v1, capital levels, benchmarks, matched controls (research period, not logged) |
| `uv run quant portfolio evaluate` | Portfolio construction grid C1-C6 (research period, one logged test per config) |
| `uv run quant audit pit` | Point-in-time and survivorship audit (descriptive, read-only) |
| `uv run quant audit masks` / `fdr` | Outcome-mask sensitivity; BY q-values next to BH (descriptive, not logged) |
| `uv run quant runs --invalidate RUN --reason TEXT` | Invalidate a run (runs are never deleted) |
| `uv run quant registry list` / `show ID` / `add` / `move` | Research Registry: every hypothesis, its state and decision (append-only) |

## Tests

```bash
uv run pytest -q                                   # full suite
uv run pytest tests/research/test_events.py -q     # one file
uv run pytest tests/test_jobs.py::test_pagination  # one test
```

## Layout

- `src/fireant_crawler/`: API client, whitelist, raw store, DuckDB warehouse, jobs, validation.
- `src/quant_research/`: feature SQL, pattern engine, statistics, backtest, daily pipeline, cross-stock,
  events.
- `src/stock_ui/`: Streamlit pages; read-only on the databases.
- `docs/`: plans, results, pre-registrations (never edited after their runs).
- `scripts/`: daily pipeline, optional launchd installer (not installed), UI launcher.

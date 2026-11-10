# Plan: FireAnt Crawler (Phase 1 - Historical Database)

<!-- type: feature -->
<!-- status: Phase 1 complete 2026-10-03 (Steps 0-5) -->

## Goal

Build a resumable, rate-limited crawler that pulls daily market data, corporate actions,
financial-report dates and reference data from `api.fireant.vn` into a local research
warehouse. This is Phase 1 of the Quant Research System.

## Decisions (2026-10-03)

| Topic | Decision | Reason |
|-------|----------|--------|
| Universe | Common stocks on HOSE, HNX, UPCoM **including delisted symbols**, plus main indices (VNINDEX, VN30, HNXINDEX, HNX30, UPINDEX; exact codes confirmed by probe) | Survivorship-bias-free research; ETFs, covered warrants, futures, bonds are out of v1 |
| History | As far back as the API serves; probe starts from 2000-01-01 (HOSE opening) | Maximise regimes covered (2008, 2011, 2018, 2022) |
| Language | **Python 3.12** managed by `uv` for crawler and research | One language for the whole system; normalizers and parsers are reused by research code |
| Rate | 2.5 requests/second, sequential, configurable via `RATE_LIMIT_RPS` | User choice (2-3 req/s) |

## Scope

In scope (v1):
- Symbol universe snapshot (listed + delisted)
- Daily quotes: OHLC, volume/value, foreign flow, order statistics, prop trading, `adjRatio`
- Corporate actions: cash dividend, stock dividend, rights issue (ex-date, record date, execution date)
- Financial-report release dates and headline numbers (`timescale-marks`)
- Fundamental snapshot per run (`sharesOutstanding`, `freeShares`, ownership), accumulated over time
- ICB industry classification and symbol-to-industry mapping
- Data validation and Parquet export

Out of scope (v1):
- Full financial statements (`full-financial-reports`): planned as v1.1
- Intraday / tick / order-book data (not in the REST spec)
- Multiple accounts or token rotation (explicitly excluded)
- Scheduling and deployment (cron is added after v1 is stable)
- Docker (deferred until deployment to a server; v1 stays Docker-ready: all config via env vars,
  data directory configurable via `DATA_DIR`, Python version pinned via `.python-version`, `uv.lock`)
- Any non-GET endpoint

## Source endpoints (from `v1.json`, Swagger 2.0)

| Job | Endpoint | Granularity |
|-----|----------|-------------|
| `universe` | `GET /instruments`, `GET /symbols/{symbol}`, `GET /symbols/search` | per run |
| `quotes` | `GET /symbols/{symbol}/historical-quotes?startDate&endDate&offset&limit` | per symbol, date-chunked |
| `index_quotes` | same as `quotes` for index symbols | per index |
| `corporate_actions` | `GET /events/search?type&startDate&endDate&offset&limit` | market-wide, date-chunked |
| `report_marks` | `GET /symbols/{symbol}/timescale-marks?startDate&endDate` | per symbol |
| `fundamental` | `GET /symbols/{symbol}/fundamental` | per symbol, snapshot |
| `icb` | `GET /icb`, `GET /icb/{industryCode}/symbols` | per run |

## Architecture

```
CLI (fireant_crawler.cli)
  -> Job runner (resumable, checkpointed)
       -> FireAnt client (GET-only whitelist, rate limiter, retry, token from env)
            -> Raw store: data/raw/... (immutable gzipped JSON, one file per response)
  -> Normalizer: raw JSON -> DuckDB tables (idempotent upserts)
  -> Validator: data-quality checks -> validation report
  -> Exporter: DuckDB -> data/parquet/*.parquet
```

Design rules:
1. **Raw first.** Every response is stored unchanged before any parsing. The warehouse can
   always be rebuilt from `data/raw/` without re-calling the API.
2. **Idempotent.** Re-running any job never creates duplicates (primary keys on every table).
3. **Resumable.** A `crawl_state` table records progress per (job, symbol, chunk). A crash or
   expired token resumes where it stopped.
4. **Point-in-time.** Every row keeps `fetched_at`. Snapshots (fundamental, adjRatio) are never
   overwritten, only appended.

## Tech stack

- Python 3.12 via `uv`; `.env` is read by our own silent loader (`dotenv.py`), never by
  `uv run --env-file`, which echoes unparseable lines including secrets
- `httpx` (HTTP), `duckdb` (warehouse + Parquet export), `pytest` (dev)
- Research code later reads `data/parquet/` or `data/warehouse.duckdb` directly

## Directory layout

```
stock/
  .env                     # FIREANT_TOKEN=... (gitignored)
  .env.example             # FIREANT_TOKEN=<paste-token-here>
  .gitignore               # .env, data/, .venv/
  .python-version
  pyproject.toml
  uv.lock
  src/fireant_crawler/
    cli.py                 # commands: probe, backfill, update, normalize, validate, export
    config.py              # env-driven settings
    client/
      fireant_client.py    # GET-only, whitelist, auth header, retry/backoff
      whitelist.py         # allowed path patterns
      rate_limiter.py
      redact.py            # token redaction
      token_info.py        # decode JWT exp/scopes locally (never prints the token)
    store/
      raw_store.py         # write/read gzipped JSON
      warehouse.py         # DuckDB connection, schema, upserts
      schema.sql
      state.py             # crawl_state checkpoints
    jobs/                  # universe, quotes, corporate_actions, report_marks, fundamental, icb
    normalize/             # quotes, report_marks (title parser), corporate_actions
    validate/checks.py
    probe.py
  tests/
    fixtures/              # copies of a.json, b.json, c.jon, d.json
    test_*.py
  data/                    # gitignored
    raw/{job}/{symbol}/{fetch_date}/{chunk}.json.gz
    warehouse.duckdb
    parquet/
    reports/
```

## Warehouse schema (DuckDB)

| Table | Primary key | Notes |
|-------|-------------|-------|
| `symbols` | `(symbol, fetched_at)` | exchange, type, isListing, icbCode; snapshot history |
| `quotes_daily` | `(symbol, date)` | raw (unadjusted) prices, all 27 fields as DOUBLE (some historical volumes are non-integer), `fetched_at` |
| `adj_ratio_segments` | `(symbol, fetched_at, start_date)` | `adjRatio` collapsed into date segments per fetch, so retroactive rewrites are visible without storing every row twice |
| `corporate_actions` | `(event_id)` | type, ex-date, record date, execution date, title, parsed ratio/amount |
| `report_marks` | `(symbol, mark_id)` | release date, period, revenue, profit, yoy text |
| `fundamental_snapshots` | `(symbol, fetched_at)` | shares outstanding, free shares, ownership ratios |
| `icb_industries` | `industry_code` | symbol-to-industry comes from `symbols.icb_code` (no per-industry calls needed) |
| `crawl_state` | `(job, symbol, chunk)` | status, attempts, last_error, updated_at |
| `crawl_log` | autoincrement | request path (no token), status, duration, bytes |

Units: prices stored as returned (thousand VND, `unit = 1000`); values stored in VND.

## Key behaviors

### Pagination and chunking
- `historical-quotes` defaults to `limit=20`. The probe finds the maximum accepted `limit`.
- Requests are chunked by calendar year, then paginated with `offset` until a short page.

### Incremental update
- For each symbol, re-fetch from `last_stored_date - 10 trading days` to today.
- If any re-fetched row's `adjRatio` differs from the stored value, a corporate action happened:
  re-fetch the symbol's full history and append new `adj_ratio_segments`.

### Rate limiting and errors
- 2.5 requests/second, sequential; configurable via `RATE_LIMIT_RPS`.
- 429 / 5xx / network errors: exponential backoff with jitter (honours `Retry-After`),
  max 5 attempts, then mark the chunk `failed`.
- 401 / 403: stop the whole run immediately with "token expired or missing scope";
  state is kept so `update` resumes after the token is refreshed.

### Safety
- The client refuses any method other than GET and any path not in `whitelist.py`
  (orders, accounts, me, admin endpoints are unreachable by construction).
- Token is read only from `FIREANT_TOKEN`; never logged, never written to raw files;
  the token string is redacted from all error output.
- The CLI prints the token's expiry and scopes (decoded locally from the JWT) at startup,
  without printing the token.

## Validation checks (from the analysis of a/b/c/d.json)

1. `low <= min(open, close)` and `high >= max(open, close)`
2. `priceAverage * totalVolume * unit ~= totalValue` (relative error < 1e-6)
3. `priceBasic == previous close`, except on known ex-dates from `corporate_actions`
4. No duplicate `(symbol, date)`; no gaps versus the trading calendar derived from VNINDEX
5. `totalVolume == dealVolume + putthroughVolume`
6. Ex-date check: on each cash-dividend ex-date, `priceBasic ~= prevClose - dividend/1000`
   (confirms that stored prices are unadjusted)
7. Report: per-symbol counts, first/last date, failed checks -> `data/reports/validation-{date}.md`

## Execution steps

### Step 0 - Skeleton + probe
Minimal project (pyproject, config, client, whitelist, rate limiter, redaction, raw store) with
unit tests, plus `uv run fireant probe`, which answers:
- Max accepted `limit` for `historical-quotes`
- Earliest available date for VOS and VNINDEX
- Exact index symbol codes
- Whether delisted symbols are served (`isListing=false`) and have quote history
- Whether prices are unadjusted: VOS around 2026-10-01 (cash dividend 900 VND ex-date)
- `/instruments` size and breakdown by exchange/type
- Token expiry, scopes, and any rate-limit response headers

Output: `data/reports/probe-{date}.md` + raw responses. **Checkpoint: review with user.**

#### Probe findings (2026-10-03, 39 requests, report: `data/reports/probe-2026-10-03.md`)

| Question | Finding | Design consequence |
|----------|---------|--------------------|
| Page size | `limit=5000` honoured (cap not reached) | Use `limit=5000`; most symbols need 1 request, VNINDEX 2 |
| History depth | VNINDEX from 2000-07-28 (6,370 rows), VOS from 2010-09-08 | Backfill from 2000-01-01 |
| Pagination | Newest first, no duplicates across pages | Offset paging is safe |
| Index codes | `VNINDEX`, `VN30`, `HNXINDEX`, `HNX30`, `UPINDEX` (no VN100/VNXALL) | Fixed index list |
| Prices | **Raw (unadjusted).** VOS 2026-10-01: basic 11.7 = prev close 12.6 - 0.9 | Store raw; adjust in research |
| `adjRatio` | **Cumulative, retroactive factor relative to the latest session**: VOS segments 1.2994 (<2011-05-16), 1.1570, 1.0769 (2025-09-18..2026-09-30), 1.0 (>=2026-10-01). `adjusted = raw / adjRatio` | Every corporate action rewrites `adjRatio` for all earlier rows: re-fetch the symbol's full history (1-2 requests) and append to `adj_ratio_segments` |
| Old ex-dates | 2011-05-16: `adjRatio` changes but `priceBasic == prev close` | Detect ex-dates from `adjRatio` change points, not from `priceBasic`; validation check 3 applies only where reliable |
| Corporate actions | `events/search` field names are shifted: `recordDate` = **ex-date (KHQ)**, `registrationDate` = record date (chot DS), `executionDate` = payment date | Map explicitly in the normalizer |
| Delisted symbols | Served (`FLC`, `KLF`, `HAI`: `isListing=false`, exchange `OTC`), but quotes are **forward-filled with zero volume** up to 2025-12-30 | Delisting date = last session with volume > 0; drop or flag filler rows |
| `isListing` | Unreliable alone (`ROS`: `isListing=true`, exchange `OTC`) | Status derived from exchange + last traded date |
| Universe | `/instruments`: 3,615 rows; stocks HSX 430, HNX 299, UPCOM 817, OTC 2; 2,059 warrants; 8 futures | `/instruments` does **not** enumerate delisted symbols; a discovery method is still needed (Step 1a) |
| Fundamental | VOS `sharesOutstanding = 140,000,000` | Snapshot daily |
| Rate limits | No rate-limit headers observed | Keep 2.5 req/s; watch for 429 |
| Request estimate | ~1,550 listed + delisted x (quotes 1 + marks 1 + fundamental 1 + symbol 1) | ~7,000-9,000 requests, about 1 hour at 2.5 req/s |

### Step 1a - Delisted symbol discovery (done 2026-10-03, 47 requests)

| Method | Result |
|--------|--------|
| `/symbols/search` (matches symbol **or company name**; `exchange`, `type`, `offset` work; `limit=5000` OK) | Union over A-Z, 0-9 = 1,611 stocks: exactly the 1,548 listed in `/instruments` + 63 OTC organisations (mostly never-listed banks/brokers). **Delisted stocks such as FLC, KLF, HAI are not returned.** |
| `/events/search` market-wide, 2000-01-01..2026-12-31, `limit=5000` (5 pages) | 24,943 events (18,481 cash dividends, 4,051 stock dividends, 2,411 rights issues), unique IDs, from 2000. **1,881 symbols, of which 406 are not currently listed**, including FLC, KLF, HAI, ROS |

Universe method for the `universe` job:
1. Listed stocks: union of `/symbols/search` over A-Z, 0-9 (36 requests, full `SymbolItem` incl. `icbCode`); cross-check with `/instruments` (1,548).
2. Historical candidates: distinct symbols in all corporate-action events minus listed (406).
3. Verify each candidate with `GET /symbols/{symbol}`; keep `type = stock`.
4. Delisting date = last session with `totalVolume > 0`.

Known gap: a delisted company that never had a cash dividend, stock dividend or rights issue is
missed. This is documented as residual survivorship bias. Optional later completeness check:
brute-force `GET /symbols/{code}` over all 3-character codes (17,576 requests, about 2 hours).

Updated request estimate: ~1,950 symbols x 4 requests ~ 8,000 requests, about 55 minutes.

### Step 1 - Warehouse and normalizers
`schema.sql`, upserts, normalizers for quotes / report marks / corporate actions / fundamental.
Unit tests use `a.json`, `b.json`, `c.jon`, `d.json` as fixtures.

#### Step 1 result (2026-10-03)
- `store/schema.sql`, `store/warehouse.py` (idempotent `INSERT OR REPLACE`, in-batch dedupe, column checks)
- Views: `symbols_latest`, `quotes_daily_adjusted` (raw / adj_ratio), `symbol_trading_span` (last traded date)
- Normalizers: quotes, adj-ratio segments, corporate actions (shifted field names mapped), report marks
  (VN number format), symbols, fundamental, ICB. Unknown quote fields fail loudly.
- 67 tests pass. Full raw probe data loads cleanly: 24,943/24,943 event titles and 94/94 marks parsed.
- Data-quality notes for the validator (Step 5): 3 cash dividends without amount; max cash 66,000 VND/share;
  35 events with payment_date < ex_date; 13 with record_date < ex_date.

### Step 2 - Jobs and CLI
`backfill --job <name> [--symbols VOS,HPG] [--from 2000-01-01]`, `update`, `normalize`.
Resumable via `crawl_state`.

#### Step 2 result (2026-10-03)
- `jobs/runner.py`: task = fetch -> raw -> ingest -> checkpoint in **one transaction**; 401/403 stops the
  run; other failures are recorded in `crawl_state` and the run continues; every request is logged
  to `crawl_log`.
- `jobs/plans.py`: `corporate_actions` (market-wide, daily), `universe` (search A-Z/0-9 + instruments +
  indices, then delisted candidates from corporate actions), `icb`, `quotes`, `report_marks`, `fundamental`.
  `update` fetches a 20-day overlap window; if any stored `adj_ratio` differs it re-fetches the full
  history; stale delisted symbols (no trades for 60 days) are skipped.
- `jobs/ingest.py` is shared by live runs and `fireant normalize`, so a rebuild reproduces the warehouse.
- Raw files are named `{chunk}__{HHMMSS}.json.gz` so same-day re-runs never overwrite. Files are created with `os.link` (never replaces); a second write in the same second gets `_001`, `_002`, ... (review of 2026-11-10).
- Known simplification: `adj_ratio_segments` from multi-page or update-window fetches may be split at
  page/window boundaries; adjacent segments with equal `adj_ratio` belong together.
- CLI: `probe`, `backfill`, `update`, `normalize [--dry-run]`, `status`. 76 tests pass.

Commands:
```bash
uv run fireant backfill --jobs corporate_actions,universe,icb
uv run fireant backfill --jobs quotes,report_marks,fundamental --symbols VOS,HPG
uv run fireant update                      # corporate_actions + quotes
uv run fireant update --jobs report_marks,fundamental
uv run fireant status                      # read-only
uv run fireant normalize --dry-run
uv run fireant migrate --dry-run           # list schema changes; `migrate` applies them
```

Write rules (added 2026-10-03): only `backfill`, `update`, `normalize` and `migrate` open the warehouse
for writing. `status`, `validate`, `export` and `migrate --dry-run` are read-only (verified: file hash
unchanged). Write commands bootstrap an empty warehouse but refuse to run while schema changes are
pending; `migrate` only adds tables/views/sequences and redefines views, and reports table drift
(changed columns) for a manual migration instead of altering tables.

### Step 3 - Pilot run
Backfill 10 symbols (VOS, HPG, FPT, VCB, VIC, VHM, MWG, SSI, HAH, GMD) + indices + 3 delisted
symbols. Run validation. **Checkpoint: review report with user.**

#### Step 3 result - pilot (2026-10-03)
Runs: `backfill --jobs corporate_actions,universe,icb` (3 min, 0 failures) and
`backfill --jobs quotes,report_marks,fundamental` for 18 symbols (2 min). The only failures were
`fundamental` for the 5 indices (HTTP 500: indices have no fundamentals); explicit `--symbols` lists
now drop indices for `report_marks`/`fundamental`. `normalize` rebuilt identical counts from raw.

| Check | Result |
|-------|--------|
| Universe | 1,546 listed stocks (HSX 430, HNX 299, UPCOM 817) + 406 delisted/OTC stocks + 55 OTC organisations + 5 indices |
| History | VNINDEX from 2000-07-28, GMD from 2002-04-22; 76,152 quote rows for 18 symbols |
| OHLC consistency | 0 violations |
| adjRatio change points vs corporate actions | **217 / 217 matched to an ex-date** in `corporate_actions` |
| priceBasic vs previous close (stocks) | equal except on ex-dates (2 exceptions, both delisted symbols) |
| Delisted | FLC, HAI last traded 2022-09-08, KLF 2023-03-10; afterwards forward-filled with zero volume to 2025-12-30 |
| Report marks | 1,134 marks, all parsed after adding labels `S` (stock dividend), `I` (rights issue) and empty revenue (`DT:  tỷ`, securities firms) |

Findings that change the validation rules (Step 5):
- **`priceAverage` is the matched-order VWAP, excluding put-through**: the identity is
  `price_average * deal_volume * unit = total_value - putthrough_value` (FPT: 6 / 4,932 rows off).
  For research, `AvgTradePrice = total_value / total_volume` mixes put-through trades; prefer `price_average`.
- **2025-07-16 is corrupt in the source for every stock**: `total_volume < deal_volume`
  (e.g. VOS 3,599,100 vs 4,007,300). Flag the date; do not trust volumes that day.
- Indices: `price_average`, value identity and `priceBasic` checks do not apply.
- A few sessions are missing versus the VNINDEX calendar (suspensions, e.g. SSI 10, HAI 10) and HNX
  indices have a few dates that HOSE does not (HOSE closed). Report, do not fail.

### Step 4 - Full backfill
Whole universe: `uv run fireant backfill --jobs quotes,report_marks,fundamental` (universe,
corporate actions and ICB are already loaded). Estimate: ~1,960 quote requests + ~1,950 marks +
~1,550 fundamental ~ 5,500 requests ~ 40 minutes at 2.5 req/s.

### Step 5 - Validate and export
`validate` and `export` to Parquet; write `docs/data-dictionary.md`.

## Success criteria

- [ ] `probe` answers every question in Step 0
- [ ] Client cannot send a non-GET or non-whitelisted request (unit test)
- [ ] Token never appears in logs, raw files or error output (unit test with a fake token)
- [ ] Normalizers reproduce the fixtures exactly (20 + 20 quote rows, 38 marks, 4 dividend rows)
- [ ] Re-running `backfill` creates zero duplicate rows
- [ ] Killing the process mid-run and re-running resumes without re-fetching completed chunks
- [ ] Pilot validation report has no unexplained failures
- [ ] Full backfill completes; validation report reviewed
- [ ] Parquet files readable from Python (`duckdb` or `pandas`)

## Testing

- All tests: `uv run pytest`
- Single file: `uv run pytest tests/test_whitelist.py`
- Single test: `uv run pytest tests/test_whitelist.py::test_rejects_orders_endpoint`
- Integration: `probe` and the pilot run against the real API (no mocks for these)

## Rollback

There is no production system; rollback concerns local data and the token.
- **Bad warehouse data:** delete `data/warehouse.duckdb` and run `fireant normalize`, which
  rebuilds it from `data/raw/` without API calls. Dry-run first: `fireant normalize --dry-run`.
- **Bad raw data from a run:** raw files are partitioned by `fetch_date`; list that partition,
  delete it, and re-run `update`.
- **Token leaked:** log out of FireAnt on all devices / change password to invalidate it; check
  linked brokerage accounts for unexpected orders.
- **Account rate-limited or blocked:** stop the crawler, lower `RATE_LIMIT_RPS`, resume later.
- Triggers: any validation check failing on more than 1% of rows; repeated 401/403; an
  unexpected schema change in raw responses.

## Risks

| Risk | Mitigation |
|------|------------|
| ToS violation / account block | Single account, GET-only, 2.5 req/s, off-peak runs; user accepts this risk |
| Token expiry mid-run | Fail fast on 401, resumable state |
| API schema change | Raw-first storage; normalizer fails loudly on unknown/missing fields |
| `adjRatio` semantics unclear | Probe ex-date check + `adj_ratio_segments` table |
| Delisted symbols not served | Probe; if absent, document survivorship bias and look for a second source |
| Token has `orders-write` scope | Whitelist makes trading endpoints unreachable; token kept in `.env` only |

## Phase 1 result (2026-10-03)

Full backfill: `backfill --jobs quotes,report_marks,fundamental`, ~5,500 requests, 0 failed tasks.

| Table | Rows |
|-------|------|
| quotes_daily | 5,465,766 (1,957 symbols incl. 5 indices; 1,928 with data) |
| corporate_actions | 24,943 |
| report_marks | 109,805 |
| fundamental_snapshots | 1,549 |
| symbols | 2,012 |
| adj_ratio_segments | 19,858 |

Storage: raw 205 MB, warehouse 385 MB, Parquet 283 MB.

Validation (`data/reports/validation-2026-10-03.md`), after refining rules with what the data showed:
- No error-level findings.
- Market rules learned: UPCOM reference price = previous **average** price (not close);
  `price_average` is often rounded to the tick.
- Residual source noise: vwap identity off on 33,963 rows (0.6%), reference-price rule on 12,682
  (0.2%), 4 OHLC anomalies in the source, 9 adj changes without a matching event (bonus shares etc.),
  2025-07-16 volume corruption, 46 events with inconsistent dates, 3 cash dividends without amount.
- 5 `failed_tasks` are the pilot's index `fundamental` requests (expected; indices have none).

Next candidates: daily `update` scheduling (cron after 18:00 VN time), v1.1 full financial statements,
then Phase 2 (feature engine) of the research plan.


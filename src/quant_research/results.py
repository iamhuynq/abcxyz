"""results.duckdb: persistent store of research runs, statistics and the hypothesis log.

Never rebuilt. Runs are invalidated, not deleted, so the multiple-testing log stays complete. Run status:
running (started) -> ok (finished) or failed (raised); invalid is set by hand with a reason.
The research database is attached READ_ONLY as `rs`.

Invalidation policy (docs/research-integrity.md, G-2): invalidate a run only for a defect established
independently of its result (a code, data or method error that the reason names), never because a result is
unwelcome. The run, its tests and the reason stay; `run_invalidations` records when (from 2026-11-11), and
`quant audit fdr` compares the family with and without invalid runs.
"""

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from fireant_crawler.store.locking import connect_with_retry

PERIODS = ("research", "validation", "holdout", "forward")

SCHEMA = """
CREATE TABLE IF NOT EXISTS pattern_definitions (
    name VARCHAR NOT NULL, version VARCHAR NOT NULL, where_sql VARCHAR NOT NULL, base VARCHAR,
    hypothesis VARCHAR, doc_ref VARCHAR, created_at TIMESTAMPTZ NOT NULL, PRIMARY KEY (name, version));
CREATE TABLE IF NOT EXISTS research_runs (
    run_id VARCHAR PRIMARY KEY, kind VARCHAR NOT NULL, pattern_name VARCHAR, pattern_version VARCHAR,
    period VARCHAR NOT NULL, params VARCHAR NOT NULL, feature_build_id VARCHAR, code_hash VARCHAR NOT NULL,
    created_at TIMESTAMPTZ NOT NULL, status VARCHAR NOT NULL, invalid_reason VARCHAR);
CREATE TABLE IF NOT EXISTS pattern_stats (
    run_id VARCHAR NOT NULL, horizon VARCHAR NOT NULL, segment VARCHAR NOT NULL,
    n_events BIGINT, n_dates BIGINT, mean DOUBLE, event_mean DOUBLE, median DOUBLE, win_rate DOUBLE,
    std DOUBLE, se DOUBLE, t DOUBLE, p DOUBLE, ci_low DOUBLE, ci_high DOUBLE, mean_after_cost DOUBLE,
    PRIMARY KEY (run_id, horizon, segment));
CREATE TABLE IF NOT EXISTS pattern_comparisons (
    run_id VARCHAR NOT NULL, horizon VARCHAR NOT NULL, base_name VARCHAR NOT NULL,
    n_dates BIGINT, diff_mean DOUBLE, se DOUBLE, t DOUBLE, p DOUBLE, PRIMARY KEY (run_id, horizon));
-- Primary test: pattern minus the whole filtered universe on the same dates (date-level lift).
CREATE TABLE IF NOT EXISTS pattern_lifts (
    run_id VARCHAR NOT NULL, horizon VARCHAR NOT NULL, n_dates BIGINT, pattern_mean DOUBLE,
    baseline_mean DOUBLE, baseline_win_rate DOUBLE, lift DOUBLE, se DOUBLE, t DOUBLE, p DOUBLE,
    PRIMARY KEY (run_id, horizon));
CREATE TABLE IF NOT EXISTS pattern_occurrences (
    run_id VARCHAR NOT NULL, symbol VARCHAR NOT NULL, date DATE NOT NULL, market_regime VARCHAR,
    exchange_now VARCHAR, out_5d DOUBLE, out_10d DOUBLE, out_20d DOUBLE, PRIMARY KEY (run_id, symbol, date));
-- Append-only record of invalidations made from 2026-11-11 on (older invalidations have no row: time unknown).
CREATE TABLE IF NOT EXISTS run_invalidations (
    run_id VARCHAR NOT NULL, invalidated_at TIMESTAMPTZ NOT NULL, reason VARCHAR NOT NULL);
CREATE TABLE IF NOT EXISTS scan_deciles (
    run_id VARCHAR NOT NULL, feature VARCHAR NOT NULL, horizon VARCHAR NOT NULL, segment VARCHAR NOT NULL,
    decile INTEGER NOT NULL, mean DOUBLE, n_dates BIGINT, n_events BIGINT,
    PRIMARY KEY (run_id, feature, horizon, segment, decile));
CREATE TABLE IF NOT EXISTS scan_stats (
    run_id VARCHAR NOT NULL, feature VARCHAR NOT NULL, horizon VARCHAR NOT NULL, segment VARCHAR NOT NULL,
    n_dates BIGINT, spread DOUBLE, spread_t DOUBLE, spread_p DOUBLE, ic_mean DOUBLE, ic_t DOUBLE, ic_p DOUBLE,
    monotonicity DOUBLE, PRIMARY KEY (run_id, feature, horizon, segment));
CREATE TABLE IF NOT EXISTS hypothesis_log (
    run_id VARCHAR NOT NULL, label VARCHAR NOT NULL, period VARCHAR NOT NULL, p_value DOUBLE,
    logged_at TIMESTAMPTZ NOT NULL, PRIMARY KEY (run_id, label));
-- BH q-values over every test of every valid run, recomputed on each read. q_by is the Benjamini-Yekutieli
-- q (BH times c(m) = 1 + 1/2 + ... + 1/m, capped at 1): a diagnostic valid under any dependence. Decisions
-- use q_value.
CREATE OR REPLACE VIEW hypothesis_q AS
WITH h AS (
    SELECT h.*, r.kind, r.pattern_name FROM hypothesis_log h JOIN research_runs r USING (run_id)
    WHERE r.status = 'ok' AND h.p_value IS NOT NULL AND NOT isnan(h.p_value)   -- NaN is not NULL in DuckDB
), ranked AS (
    SELECT *, row_number() OVER (ORDER BY p_value, run_id, label) AS rk, count(*) OVER () AS m FROM h
), adjusted AS (
    SELECT *, min(p_value * m / rk) OVER (ORDER BY rk DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS bh,
           sum(1.0 / rk) OVER () AS c_m
    FROM ranked
)
SELECT run_id, label, period, kind, pattern_name, p_value, rk, m,
       least(1.0, bh) AS q_value, least(1.0, bh * c_m) AS q_by
FROM adjusted;
"""


def stored_params(params: "ResearchParams") -> dict[str, Any]:
    """Parameters stored with a run. The mask settings appear only when they differ from the recorded
    definition, so runs on the defaults store exactly what they stored before the masks were split out."""
    out = asdict(params)
    if tuple(out["outcome_masks"]) == OUTCOME_MASKS and out["mask_window"] == "legacy20":
        del out["outcome_masks"], out["mask_window"]
    return out


class HoldoutLockedError(RuntimeError):
    pass


OUTCOME_MASKS = ("quality", "execution", "period")
MASK_HORIZONS = (1, 3, 5, 10, 20)


@dataclass(frozen=True)
class ResearchParams:
    """Universe filter and test settings (stored with every run)."""
    min_adv_value: float = 1e9
    min_session_index: int = 20
    min_stocks_per_date: int = 30      # for cross-sectional deciles / IC
    cost_round_trip: float = 0.004
    horizons: tuple[int, ...] = (5, 10, 20)
    extra: dict[str, Any] = field(default_factory=dict)
    # Outcome masks applied on top of the point-in-time universe (review of 2026-11-10). The defaults are the
    # recorded definition: all three masks over the legacy 20-session window. "horizon" uses each target's
    # own window. Stored with a run only when they differ from the defaults (see stored_params).
    outcome_masks: tuple[str, ...] = OUTCOME_MASKS
    mask_window: str = "legacy20"

    def __post_init__(self) -> None:
        unknown = set(self.outcome_masks) - set(OUTCOME_MASKS)
        if unknown or self.mask_window not in ("legacy20", "horizon"):
            raise ValueError(f"Unknown outcome masks {sorted(unknown)} or mask window {self.mask_window!r}")

    def point_in_time_sql(self) -> str:
        """Universe known at the close of t: no outcome information."""
        return (f"is_traded AND adv_value_20 > {self.min_adv_value} AND session_index > {self.min_session_index}"
                " AND NOT price_jump AND NOT bad_source_date")

    def mask_sql(self, target: str | None = None) -> str:
        """The chosen outcome masks for one target column ("TRUE" when none). The horizon window takes the
        horizon from the target's `_{h}d` suffix; a loader that reads several horizons at once (target None)
        masks with the longest of `horizons`. The legacy window ignores the target."""
        h = None
        if self.mask_window == "horizon":
            if target is None:
                h = max(self.horizons)
            else:
                m = re.search(r"_(\d+)d$", target)
                h = int(m.group(1)) if m else None
            if h not in MASK_HORIZONS:
                raise ValueError(f"No per-horizon mask for target {target!r} (horizons {self.horizons})")
        parts = []
        for mask in OUTCOME_MASKS:                  # fixed order: the SQL text does not depend on tuple order
            if mask not in self.outcome_masks:
                continue
            if mask == "quality":
                parts.append(f"NOT coalesce(fwd_has_price_jump_{h or 20}d, false)")
            elif mask == "execution":
                parts.append("NOT coalesce(entry_blocked, false)")
            else:
                parts.append(f"NOT coalesce({'crosses_period' if h is None else f'crosses_period_{h}d'}, false)")
        return " AND ".join(parts) or "TRUE"

    def sample_sql(self, target: str | None = None) -> str:
        """Rows a research statistic uses: the point-in-time universe and the outcome masks."""
        return f"{self.point_in_time_sql()} AND {self.mask_sql(target)}"

    def universe_sql(self) -> str:
        """The recorded definition (all masks, legacy window); kept for the daily report and old callers."""
        return (f"is_traded AND adv_value_20 > {self.min_adv_value} AND session_index > {self.min_session_index}"
                " AND NOT price_jump AND NOT coalesce(fwd_has_price_jump_20d, false) AND NOT bad_source_date"
                " AND NOT coalesce(entry_blocked, false) AND NOT coalesce(crosses_period, false)")


def research_code_files(root: Path | None = None) -> list[Path]:
    """Every .py and .sql file of the quant_research package, subpackages included (backtest/, cross/,
    portfolio/): all the code that can change a research result."""
    root = root or Path(str(resources.files("quant_research")))
    return [p for p in sorted(root.rglob("*")) if p.suffix in (".py", ".sql") and "__pycache__" not in p.parts]


def research_code_hash(root: Path | None = None) -> str:
    """Hash of research_code_files (relative paths and bytes); stored with every run as research_runs.code_hash
    and provenance["code_hash"]. Not the same as the feature-build hash (quant_research.build)."""
    root = root or Path(str(resources.files("quant_research")))
    digest = hashlib.sha256()
    for p in research_code_files(root):
        digest.update(str(p.relative_to(root)).encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()[:16]


def check_period(period: str, final: bool) -> None:
    if period not in PERIODS:
        raise ValueError(f"Unknown period {period!r}; use one of {PERIODS}")
    if period == "holdout" and not final:
        raise HoldoutLockedError("The holdout period is locked. Use --final only for the final, pre-registered test.")


class ResultsStore:
    def __init__(self, results_path: Path, research_path: Path) -> None:
        if not research_path.exists():
            raise FileNotFoundError(f"No research database at {research_path}; run `quant build`")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = connect_with_retry(results_path)  # the local UI may be reading
        self.con.execute(SCHEMA)
        from quant_research import registry
        try:
            registry.ensure(self.con)
        except Exception:
            self.con.close()
            raise
        self.con.execute(f"ATTACH '{research_path.as_posix().replace(chr(39), chr(39) * 2)}' AS rs (READ_ONLY)")
        self._universe_cache: dict[tuple, tuple] = {}
        self._open_runs: set[str] = set()

    def close(self) -> None:
        self._fail_open_runs("not finished: the store closed before finish_run")
        self.con.execute("DETACH rs")
        self.con.close()

    def __enter__(self) -> "ResultsStore":
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        if exc_type is not None:
            self._fail_open_runs(f"{exc_type.__name__}: {exc}"[:500])
        self.close()

    def _fail_open_runs(self, reason: str) -> None:
        for run_id in sorted(self._open_runs):                       # only runs still running: never overwrite
            self.con.execute("""UPDATE research_runs SET status = 'failed', invalid_reason = ?
                                WHERE run_id = ? AND status = 'running'""", [reason, run_id])
        self._open_runs.clear()

    def feature_build_id(self) -> str | None:
        row = self.con.execute("SELECT build_id FROM rs.feature_builds ORDER BY built_at DESC LIMIT 1").fetchone()
        return row[0] if row else None

    def start_run(self, run_id: str, kind: str, period: str, params: ResearchParams, now: datetime,
                  pattern: tuple[str, str] | None = None, config: dict | None = None) -> None:
        """`config`: an engine's own configuration (the ML experiment), stored under "config" and covered by the
        config hash. Without it the stored row and the hash are exactly as before."""
        base = stored_params(params)
        if config is not None:
            if not isinstance(config, dict) or "config" in base:
                raise ValueError("config must be a dict")
            base = {**base, "config": config}
        config_json = json.dumps(base, default=str, sort_keys=True)
        stored = {**base, "provenance": self.provenance(config_json)}
        self.con.execute(
            "INSERT INTO research_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', NULL)",
            [run_id, kind, pattern[0] if pattern else None, pattern[1] if pattern else None, period,
             json.dumps(stored, default=str), self.feature_build_id(), research_code_hash(), now],
        )
        self._open_runs.add(run_id)

    def finish_run(self, run_id: str) -> None:
        """running -> ok, once every result and hypothesis row of the run is written. A run that raises is set to
        failed when the store exits; a process killed mid-run leaves `running`. Only `ok` runs count anywhere
        (q-values, duplicate guards, reports)."""
        changed = self.con.execute("UPDATE research_runs SET status = 'ok' WHERE run_id = ? AND status = 'running'",
                                   [run_id]).fetchone()[0]
        if changed != 1:
            row = self.con.execute("SELECT status FROM research_runs WHERE run_id = ?", [run_id]).fetchone()
            state = "missing" if row is None else f"in state {row[0]!r}"
            raise RuntimeError(f"finish_run({run_id}): the run is {state}, not 'running'; nothing was marked ok")
        self._open_runs.discard(run_id)

    def stuck_runs(self) -> list[tuple[str, str]]:
        """Runs left `running` by a process that was killed: not valid anywhere. Recovery: inspect, then
        `quant runs --invalidate RUN --reason "process killed"` (runs are never deleted) and run again."""
        return self.con.execute("""SELECT run_id, created_at::VARCHAR FROM research_runs WHERE status = 'running'
                                   ORDER BY created_at""").fetchall()

    def provenance(self, config_json: str | None = None) -> dict:
        from quant_research.provenance import collect
        return collect(self.con, research_code_hash(), config_json)

    def log_hypothesis(self, run_id: str, label: str, period: str, p_value: float | None, now: datetime) -> None:
        """Log one test. A p-value that could not be computed (None or NaN) is stored as NULL: the test stays
        on record but is outside the FDR family. A p-value outside [0, 1] is a bug and is refused."""
        if p_value is not None and math.isnan(p_value):
            p_value = None
        if p_value is not None and not 0.0 <= p_value <= 1.0:
            raise ValueError(f"p-value {p_value!r} of {run_id}/{label} is outside [0, 1]")
        self.con.execute("INSERT OR REPLACE INTO hypothesis_log VALUES (?, ?, ?, ?, ?)",
                         [run_id, label, period, p_value, now])

    def invalidate(self, run_id: str, reason: str, now: datetime | None = None) -> int:
        """Mark a run invalid (see the invalidation policy in the module docstring) and record when. Returns 0 for
        an unknown run and for a run that is already invalid (its first reason is kept, nothing is recorded)."""
        cur = self.con.execute("""UPDATE research_runs SET status = 'invalid', invalid_reason = ?
                                  WHERE run_id = ? AND status <> 'invalid'""", [reason, run_id])  # reason kept once set
        changed = cur.fetchone()[0]
        if changed:
            self._open_runs.discard(run_id)                          # an invalid run is closed: never 'failed' later
            self.con.execute("INSERT INTO run_invalidations VALUES (?, ?, ?)",
                             [run_id, now or datetime.now(UTC), reason])
        return changed

    def q_values(self, run_id: str) -> dict[str, float]:
        return dict(self.con.execute("SELECT label, q_value FROM hypothesis_q WHERE run_id = ?", [run_id]).fetchall())

    def universe_by_date(self, period: str, params: "ResearchParams", column: str) -> tuple:
        """(date ints, mean outcome per date, win rate) of the filtered universe; cached per run batch."""
        key = (period, params.sample_sql(column), column)
        cache = self._universe_cache
        if key not in cache:
            rows = self.con.execute(f"""SELECT date, avg({column}) AS m, avg(({column} > 0)::INT) AS w
                FROM rs.feature_target WHERE {params.sample_sql(column)} AND period = ? AND {column} IS NOT NULL
                GROUP BY date ORDER BY date""", [period]).fetchall()
            dates = np.array([r[0].toordinal() for r in rows], dtype=np.int64) - 719163  # days since 1970-01-01
            cache[key] = (dates, np.array([r[1] for r in rows], dtype=float), float(np.mean([r[2] for r in rows])) if rows else None)
        return cache[key]

    def holdout_runs(self) -> int:
        return self.con.execute("SELECT count(*) FROM research_runs WHERE period = 'holdout'").fetchone()[0]

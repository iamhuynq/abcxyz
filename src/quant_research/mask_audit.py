"""Outcome-mask sensitivity audit (docs/review-2026-11-10-fixes-plan.md, fix 2). Research period only.

Every recorded research-period analysis that reads the research sample (library patterns, the decile scan,
the interaction scan, the event study and the cross-stock tests) is re-run once per mask configuration.
Each configuration runs in its own scratch results database, so the audit logs no hypothesis in the real
store: the multiple-testing families and their q-values do not change. q-values shown by the audit are
computed within each configuration's own scratch family, which has the same tests in every configuration.

Declared rule (before running): a test is mask-sensitive when, against the combined configuration (the
recorded definition), its effect changes sign or its q < 0.05 status flips. Validated, failed and forward
hypotheses are not re-decided; their robustness is reported only.
"""

import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pandas as pd

from quant_research.cross.hypotheses import TESTS, attach, run_all
from quant_research.cross.params import CrossParams
from quant_research.engine import run_pattern
from quant_research.engine import run_scan as run_decile_scan
from quant_research.event_study import run_event_study
from quant_research.interaction_report import is_candidate
from quant_research.interactions import run_scan as run_interaction_scan
from quant_research.patterns import LIBRARY
from quant_research.results import OUTCOME_MASKS, ResearchParams, ResultsStore

PERIOD = "research"
REFERENCE = "combined"
CONFIGS: dict[str, dict] = {
    "baseline": {"outcome_masks": ()},
    "quality": {"outcome_masks": ("quality",)},
    "execution": {"outcome_masks": ("execution",)},
    "period": {"outcome_masks": ("period",)},
    "combined": {},
    "combined_horizon": {"mask_window": "horizon"},
}
TARGETS = ("fwd_excess_exec_5d", "fwd_excess_exec_10d", "fwd_excess_exec_20d")
Q_MAX = 0.05


def run_config(path: Path, research: Path, cross: Path | None, params: ResearchParams, now: datetime,
               log: Callable[[str], None]) -> None:
    """All recorded research-period analyses on one mask configuration, into a scratch results file."""
    with ResultsStore(path, research) as store:
        for pattern in LIBRARY.values():
            run_pattern(store, pattern, PERIOD, params, now)
        log("  patterns done")
        run_decile_scan(store, PERIOD, params, now)
        log("  decile scan done")
        run_interaction_scan(store, PERIOD, now, params)
        run_event_study(store, now, params)
        log("  interactions and event study done")
        if cross is not None:
            attach(store, cross)
            run_all(store, PERIOD, params, now, CrossParams(), tests=TESTS, log=lambda _m: None)
            log("  cross tests done")


def run_audit(research: Path, cross: Path | None, now: datetime, log: Callable[[str], None] = print,
              configs: dict[str, dict] = CONFIGS) -> dict[str, dict[str, pd.DataFrame]]:
    """Runs every configuration in a temporary directory and returns the collected tables per configuration."""
    out = {}
    with tempfile.TemporaryDirectory(prefix="mask-audit-") as tmp:
        for name, kwargs in configs.items():
            log(f"configuration {name}")
            path = Path(tmp) / f"{name}.duckdb"
            run_config(path, research, cross, ResearchParams(**kwargs), now, log)
            with ResultsStore(path, research) as store:
                out[name] = collect(store)
                out[name]["sample"] = sample_stats(store, ResearchParams(**kwargs))
    return out


def collect(store: ResultsStore) -> dict[str, pd.DataFrame]:
    con = store.con
    q = "LEFT JOIN hypothesis_q q ON q.run_id = x.run_id AND q.label = "
    tables = {
        "patterns": con.execute(f"""
            SELECT r.pattern_name AS name, x.horizon, x.n_dates, x.lift AS effect, x.t, x.p, q.q_value AS q
            FROM pattern_lifts x JOIN research_runs r USING (run_id) {q} 'lift_' || x.horizon""").df(),
        "pattern_events": con.execute("""
            SELECT r.pattern_name AS name, s.horizon, s.n_events FROM pattern_stats s JOIN research_runs r
            USING (run_id) WHERE s.segment = 'all'""").df(),
        "scan": con.execute("""
            SELECT feature AS name, horizon, n_dates, ic_mean AS effect, ic_t AS t, ic_p AS p, spread
            FROM scan_stats WHERE segment = 'all'""").df(),
        "events": con.execute("""
            SELECT event_type AS name, horizon, n_events, lift AS effect, lift_t AS t
            FROM event_study_stats WHERE segment = 'all' AND lift IS NOT NULL""").df(),
    }
    ix = con.execute(f"""
        SELECT x.factor || ' x ' || x.dimension AS name, x.diff, x.t, x.p, q.q_value,
               x.diff_first_half, x.diff_second_half
        FROM interaction_tests x {q} 'ix_' || x.factor || '_' || x.dimension""").df()
    ix["candidate"] = ix.apply(is_candidate, axis=1) if len(ix) else []
    tables["interactions"] = ix.assign(effect=ix["diff"]).rename(columns={"q_value": "q"})
    has_cross = con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'cross_stats'"
                            ).fetchone()[0]
    tables["cross"] = con.execute(f"""
        SELECT x.test AS name, x.horizon, x.n, x.estimate AS effect, x.t, x.p
        FROM cross_stats x""").df() if has_cross else pd.DataFrame()
    return tables


def sample_stats(store: ResultsStore, params: ResearchParams) -> pd.DataFrame:
    """Rows kept and the outcome distribution per target."""
    rows = []
    for y in TARGETS:
        r = store.con.execute(f"""
            SELECT count(*), avg({y}), quantile_cont({y}, [0.05, 0.5, 0.95])
            FROM rs.feature_target WHERE {params.sample_sql(y)} AND period = ? AND {y} IS NOT NULL""",
                              [PERIOD]).fetchone()
        rows.append({"target": y, "rows": r[0], "mean": r[1], "p05": r[2][0], "p50": r[2][1], "p95": r[2][2]})
    return pd.DataFrame(rows)


def removed_by_year(store: ResultsStore, params: ResearchParams = ResearchParams()) -> pd.DataFrame:
    """Share of point-in-time rows each mask removes, per year (10-session target)."""
    y = "fwd_excess_exec_10d"
    cols = ", ".join(f"avg(CASE WHEN {ResearchParams(outcome_masks=(m,)).mask_sql(y)} THEN 0 ELSE 1 END) AS {m}"
                     for m in OUTCOME_MASKS)
    return store.con.execute(f"""
        SELECT year(date) AS year, count(*) AS rows, {cols} FROM rs.feature_target
        WHERE {params.point_in_time_sql()} AND period = ? AND {y} IS NOT NULL GROUP BY 1 ORDER BY 1""",
                             [PERIOD]).df()


def concentration(store: ResultsStore, params: ResearchParams = ResearchParams()) -> pd.DataFrame:
    """Per library pattern: share of its point-in-time hits each mask removes, next to the overall share.
    A small overall share can still sit inside one pattern."""
    y = "fwd_excess_exec_10d"
    flags = {m: f"CASE WHEN {ResearchParams(outcome_masks=(m,)).mask_sql(y)} THEN 0 ELSE 1 END" for m in OUTCOME_MASKS}
    sel = ", ".join(f"avg({f}) AS {m}" for m, f in flags.items())
    overall = store.con.execute(f"""SELECT 'all rows' AS name, count(*) AS hits, {sel} FROM rs.feature_target
        WHERE {params.point_in_time_sql()} AND period = ?""", [PERIOD]).df()
    parts = [overall]
    for p in LIBRARY.values():
        src = p.source_sql(params.point_in_time_sql(), params.min_stocks_per_date)
        df = store.con.execute(f"SELECT ? AS name, count(*) AS hits, {sel} FROM ({src})", [p.name, PERIOD]).df()
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def compare(results: dict[str, dict[str, pd.DataFrame]], table: str, keys: tuple[str, ...]) -> pd.DataFrame:
    """One row per test: the reference effect and, per configuration, the effect and whether the declared
    rule marks it mask-sensitive (sign change, or q / p < 0.05 status flip when q is not available)."""
    ref = results[REFERENCE][table]
    if ref.empty:
        return pd.DataFrame()
    level = "q" if "q" in ref.columns else ("p" if "p" in ref.columns else None)
    base = ref[[*keys, "effect", *([level] if level else [])]].rename(
        columns={"effect": "ref_effect", **({level: "ref_sig"} if level else {})})
    for name, tables in results.items():
        if name == REFERENCE or tables[table].empty:
            continue
        cur = tables[table][[*keys, "effect", *([level] if level else [])]].rename(
            columns={"effect": f"{name}_effect", **({level: f"{name}_sig"} if level else {})})
        base = base.merge(cur, on=list(keys), how="outer")
        flip_sign = (base["ref_effect"] > 0) != (base[f"{name}_effect"] > 0)
        flip_sig = ((base["ref_sig"] < Q_MAX) != (base[f"{name}_sig"] < Q_MAX)) if level else False
        both = base["ref_effect"].notna() & base[f"{name}_effect"].notna() & (base["ref_effect"] != 0)
        base[f"{name}_sensitive"] = both & (flip_sign | flip_sig)
    base.attrs["level"] = level
    return base

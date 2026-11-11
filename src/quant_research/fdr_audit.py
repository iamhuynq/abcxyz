"""`quant audit fdr`: Benjamini-Yekutieli q-values next to the recorded Benjamini-Hochberg ones.

Descriptive (docs/contract-and-fdr-plan.md, part B). BH controls the false discovery rate under independence or
positive dependence; our tests overlap (same stocks, dates and horizons), so the report shows which findings
also pass BY, which is valid under any dependence. Nothing is logged; no registry state changes. A finding that
passes BH but not BY is "not robust to arbitrary dependence"; any state change goes through
`quant registry move`.
"""

from datetime import datetime

from quant_research import provenance
from quant_research.results import ResultsStore, research_code_hash

Q_MAX = 0.05


def _q(v) -> str:
    return "-" if v is None else f"{v:.3g}"


def render(store: ResultsStore, now: datetime) -> str:
    con = store.con
    m, c_m = con.execute("SELECT max(m), sum(1.0 / rk) FROM hypothesis_q").fetchone()
    tests = con.execute(f"""
        SELECT run_id, label, kind, coalesce(pattern_name, ''), period, p_value, q_value, q_by
        FROM hypothesis_q WHERE q_value < {Q_MAX} ORDER BY q_value, run_id, label""").fetchall()
    survive = sum(1 for t in tests if t[7] < Q_MAX)
    lines = [f"# Multiple-testing audit: BY next to BH ({now:%Y-%m-%d})", "",
             provenance.line(provenance.collect(con, research_code_hash())), "",
             "Descriptive (docs/contract-and-fdr-plan.md, part B). Decisions use the BH q (`q_value`); BY is a "
             "diagnostic valid under any dependence between tests. A finding that passes BH but not BY is not "
             "robust to arbitrary dependence; its registry state does not change here. Nothing is logged.", "",
             f"- Family: every logged test of every valid run, m = {m or 0:,}; c(m) = {c_m or 0:.3f}. This is the "
             "global exploratory family of the whole research log (patterns, scans, cross, interaction and "
             "portfolio tests). It is not a pre-registered family: a pre-registration declares its own family "
             "before the run, and families are never split after the fact to improve q-values.",
             f"- Tests with BH q < {Q_MAX}: {len(tests):,}; of these, BY q < {Q_MAX}: {survive:,}.", "",
             f"## 1. Tests with BH q < {Q_MAX}", "",
             "| run | label | kind | pattern | period | p | q BH | q BY | passes BY |", "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r} | {lab} | {k} | {pat} | {per} | {p:.3g} | {_q(q)} | {_q(qb)} | {'yes' if qb < Q_MAX else 'no'} |"
              for r, lab, k, pat, per, p, q, qb in tests]
    rows = con.execute("""
        WITH links AS (
            SELECT hypothesis_id, run_id FROM hypothesis_transitions WHERE run_id IS NOT NULL
            UNION
            SELECT s.hypothesis_id, r.run_id FROM hypothesis_status s
            JOIN research_runs r ON r.pattern_name = s.pattern AND r.pattern_name IS NOT NULL
        )
        , own AS (   -- the tests of each hypothesis, ranked: smallest BH q, then run, then label (one row)
            SELECT s.hypothesis_id, s.family, s.status, coalesce(s.decision, '-') AS decision,
                   q.run_id, q.label, q.q_value, q.q_by,
                   count(DISTINCT q.run_id) OVER (PARTITION BY s.hypothesis_id) AS n_runs,
                   row_number() OVER (PARTITION BY s.hypothesis_id
                                      ORDER BY q.q_value NULLS LAST, q.run_id, q.label) AS pick
            FROM hypothesis_status s
            LEFT JOIN links l USING (hypothesis_id)
            LEFT JOIN hypothesis_q q ON q.run_id = l.run_id
                 AND (s.family <> 'interaction' OR q.label = 'ix_' || replace(substr(s.hypothesis_id, 7), '-', '_'))
        )
        SELECT hypothesis_id, family, status, decision, n_runs, label, q_value, q_by
        FROM own WHERE pick = 1 ORDER BY hypothesis_id""").fetchall()
    lines += ["", "## 2. Registry hypotheses", "",
              "The tests of a hypothesis: for an interaction (`P7-IX-<factor>-<dimension>`), its own test "
              "`ix_<factor>_<dimension>` of the shared scan run; otherwise every logged test of its own runs (those "
              "`quant registry show` lists: runs named in its transitions, or runs of its pattern), and the table "
              "shows the one with the smallest BH q (ties: first by run, then label). A hypothesis without logged tests shows `-`. The recorded "
              "decision used its own pre-declared rule, which this table does not re-apply.", "",
              "| hypothesis | family | state | decision | runs with tests | test | q BH | q BY | fails BY at 0.05 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for hid, fam, status, decision, n_runs, label, q, qb in rows:
        flip = "yes" if q is not None and q < Q_MAX <= qb else ""
        lines.append(f"| {hid} | {fam} | {status} | {decision} | {n_runs} | {label or '-'} | {_q(q)} | {_q(qb)} | "
                     f"{flip} |")
    return "\n".join(lines + _family_changes(con)) + "\n"


ALL_ATTEMPTS_SQL = """
    -- BH and BY over every logged test with a p-value, whatever the run status (invalid runs included)
    WITH h AS (SELECT run_id, label, p_value FROM hypothesis_log WHERE p_value IS NOT NULL AND NOT isnan(p_value)),
    ranked AS (SELECT *, row_number() OVER (ORDER BY p_value, run_id, label) AS rk, count(*) OVER () AS m FROM h),
    adjusted AS (SELECT *, min(p_value * m / rk) OVER (ORDER BY rk DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                           AS bh, sum(1.0 / rk) OVER () AS c_m FROM ranked)
    SELECT run_id, label, m, least(1.0, bh) AS q_all, least(1.0, bh * c_m) AS q_by_all FROM adjusted"""


def _family_changes(con) -> list[str]:
    """Section 3: logged tests outside the family, the family of all attempts, and each invalidated run's re-run.

    Invalidation removes a run's tests from m after its results were seen. It is acceptable only for a defect
    found independently of the result, and when the same test was re-run (a successor: an 'ok' run with the same
    name, created later), so the family keeps one copy of each test. The all-attempts family shows whether any
    verdict depends on leaving invalid runs out. A run id is `<timestamp>-<kind>-<name>-<period>`; the name is the
    id without the timestamp."""
    status = con.execute("""
        SELECT r.status, count(*), count(*) FILTER (WHERE h.p_value IS NULL OR isnan(h.p_value))
        FROM hypothesis_log h JOIN research_runs r USING (run_id) GROUP BY 1 ORDER BY 1""").fetchall()
    m_all, = con.execute(f"SELECT coalesce(max(m), 0) FROM ({ALL_ATTEMPTS_SQL})").fetchone()
    counts = con.execute(f"""
        SELECT count(*) FILTER (WHERE q.q_value < {Q_MAX}), count(*) FILTER (WHERE a.q_all < {Q_MAX}),
               count(*) FILTER (WHERE q.q_by < {Q_MAX}), count(*) FILTER (WHERE a.q_by_all < {Q_MAX})
        FROM hypothesis_q q JOIN ({ALL_ATTEMPTS_SQL}) a USING (run_id, label)""").fetchone()
    flips = con.execute(f"""
        SELECT q.run_id, q.label, q.q_value, a.q_all, q.q_by, a.q_by_all
        FROM hypothesis_q q JOIN ({ALL_ATTEMPTS_SQL}) a USING (run_id, label)
        WHERE (q.q_value < {Q_MAX}) <> (a.q_all < {Q_MAX}) OR (q.q_by < {Q_MAX}) <> (a.q_by_all < {Q_MAX})
        ORDER BY q.q_value, q.run_id, q.label""").fetchall()
    runs = con.execute(f"""
        WITH r AS (SELECT run_id, status, created_at, regexp_replace(run_id, '^[0-9T]+-', '') AS name
                   FROM research_runs),
        t AS (SELECT run_id, count(*) AS n, count(*) FILTER (WHERE p_value < {Q_MAX}) AS n_sig, min(p_value) AS min_p
              FROM hypothesis_log GROUP BY 1),
        i AS (SELECT i.run_id, t.n, t.n_sig, t.min_p,
                     (SELECT min(s.run_id) FROM r s WHERE s.name = i.name AND s.status = 'ok'
                      AND s.created_at > i.created_at) AS succ
              FROM r i JOIN t USING (run_id) WHERE i.status = 'invalid')
        SELECT i.run_id, i.n, i.n_sig, i.min_p, i.succ, s.n_sig, s.min_p
        FROM i LEFT JOIN t s ON s.run_id = i.succ ORDER BY 1""").fetchall()
    missing = sum(1 for row in runs if row[4] is None)
    out = ["", "## 3. Logged tests outside the family", "",
           "The family holds the tests with a p-value of runs whose status is `ok`. A NULL p-value is a test that "
           "could not be computed (stored on record, outside the family). A run is invalidated only for a defect "
           "found independently of its result; the re-run columns below show which way each re-run went.", "",
           "| run status | logged tests | without p-value |", "|---|---|---|"]
    out += [f"| {s} | {n:,} | {n_null:,} |" for s, n, n_null in status]
    out += ["", f"All-attempts family (every logged test with a p-value, invalid runs included): m = {m_all:,}. "
            f"Tests of `ok` runs passing at {Q_MAX}: BH {counts[0]:,} in the family, {counts[1]:,} with all attempts; "
            f"BY {counts[2]:,} and {counts[3]:,}. Tests whose verdict differs between the two: {len(flips)}."]
    if flips:
        out += ["", "| run | label | q BH | q BH all | q BY | q BY all |", "|---|---|---|---|---|---|"]
        out += [f"| {r} | {lab} | {_q(q)} | {_q(qa)} | {_q(qb)} | {_q(qba)} |" for r, lab, q, qa, qb, qba in flips]
    out += ["", f"Invalidated runs with logged tests: {len(runs)}; without a later `ok` run of the same name: "
            f"{missing}.", "",
            f"| invalidated run | tests | p < {Q_MAX} | min p | re-run as | re-run p < {Q_MAX} | re-run min p |",
            "|---|---|---|---|---|---|---|"]
    out += [f"| {r} | {n} | {k} | {_q(mp)} | {succ or 'none'} | {'-' if k2 is None else k2} | {_q(mp2)} |"
            for r, n, k, mp, succ, k2, mp2 in runs]
    return out + _provenance(con)


def _provenance(con) -> list[str]:
    """Section 4: what the family's runs recorded about how they were produced (read-only)."""
    row = con.execute("""
        WITH r AS (SELECT run_id, try_cast(params AS JSON) -> 'provenance' AS p FROM research_runs
                   WHERE status = 'ok' AND run_id IN (SELECT DISTINCT run_id FROM hypothesis_q))
        SELECT count(*), count(p), count(p ->> 'warehouse_sha'), count(p ->> 'config_hash'), count(p ->> 'env'),
               count(p ->> 'selection'), count(DISTINCT p ->> 'git_sha'),
               count(*) FILTER (WHERE (p ->> 'git_dirty') = 'true')
        FROM r""").fetchone()
    n, prov, wh, cfg, env, sel, commits, dirty = row
    return ["", "## 4. Provenance of the family's runs", "",
            "Recorded with each `ok` run that has tests in the family. Runs made before a field existed do not have "
            "it; nothing is filled in afterwards. This table checks what was recorded; it does not re-run anything.",
            "", "| runs | with provenance | warehouse hash | config hash | environment | selection rule | "
            "distinct commits | uncommitted changes |", "|---|---|---|---|---|---|---|---|",
            f"| {n} | {prov} | {wh} | {cfg} | {env} | {sel} | {commits} | {dirty} |"]

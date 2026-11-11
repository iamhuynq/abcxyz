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
             f"- Family: every logged test of every valid run, m = {m or 0:,}; c(m) = {c_m or 0:.3f}.",
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
        SELECT s.hypothesis_id, s.family, s.status, coalesce(s.decision, '-'), count(DISTINCT q.run_id),
               arg_min(q.label, q.q_value), min(q.q_value), arg_min(q.q_by, q.q_value)
        FROM hypothesis_status s
        LEFT JOIN links l USING (hypothesis_id)
        LEFT JOIN hypothesis_q q ON q.run_id = l.run_id
             AND (s.family <> 'interaction' OR q.label = 'ix_' || replace(substr(s.hypothesis_id, 7), '-', '_'))
        GROUP BY ALL ORDER BY s.hypothesis_id""").fetchall()
    lines += ["", "## 2. Registry hypotheses", "",
              "The tests of a hypothesis: for an interaction (`P7-IX-<factor>-<dimension>`), its own test "
              "`ix_<factor>_<dimension>` of the shared scan run; otherwise every logged test of its own runs (those "
              "`quant registry show` lists: runs named in its transitions, or runs of its pattern), and the table "
              "shows the one with the smallest BH q. A hypothesis without logged tests shows `-`. The recorded "
              "decision used its own pre-declared rule, which this table does not re-apply.", "",
              "| hypothesis | family | state | decision | runs with tests | test | q BH | q BY | fails BY at 0.05 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for hid, fam, status, decision, n_runs, label, q, qb in rows:
        flip = "yes" if q is not None and q < Q_MAX <= qb else ""
        lines.append(f"| {hid} | {fam} | {status} | {decision} | {n_runs} | {label or '-'} | {_q(q)} | {_q(qb)} | "
                     f"{flip} |")
    return "\n".join(lines) + "\n"

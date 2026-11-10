"""Report of the outcome-mask sensitivity audit (quant_research.mask_audit). Descriptive: nothing is logged."""

from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from quant_research import mask_audit as ma
from quant_research import provenance
from quant_research.results import research_code_hash

OTHERS = [c for c in ma.CONFIGS if c != ma.REFERENCE]
SECTIONS = (  # table, keys, title
    ("patterns", ("name", "horizon"), "Library and cross patterns: lift vs the universe (q within the configuration)"),
    ("scan", ("name", "horizon"), "Decile scan: mean IC (p of the IC t)"),
    ("interactions", ("name",), "Interaction scan: IC difference A - B (q within the configuration)"),
    ("events", ("name", "horizon"), "Event study: lift vs the universe (no p: descriptive)"),
    ("cross", ("name", "horizon"), "Cross-stock tests: estimate (p)"),
)


def _fmt(v, spec: str) -> str:
    return "-" if v is None or pd.isna(v) else format(v, spec)


def _md(df: pd.DataFrame, specs: dict[str, str]) -> list[str]:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c], specs[c]) if c in specs else str(r[c]) for c in df.columns) + " |")
    return lines


def summary(results: dict) -> pd.DataFrame:
    """Per table and configuration: tests compared, statistically sensitive tests and (in brackets) material ones."""
    rows = []
    for table, keys, _ in SECTIONS:
        cmp = ma.compare(results, table, keys)
        if cmp.empty:
            continue
        row = {"analysis": table, "tests": int(cmp["ref_effect"].notna().sum())}
        for name in OTHERS:
            col, mat = f"{name}_sensitive", f"{name}_material"
            row[name] = f"{int(cmp[col].sum())} ({int(cmp[mat].sum())})" if col in cmp else None
        rows.append(row)
    return pd.DataFrame(rows)


def ranks(results: dict) -> pd.DataFrame:
    """Spearman agreement of test statistics with the reference, per table (1.0 = same ranking)."""
    rows = []
    for table, keys, _ in SECTIONS:
        ref = results[ma.REFERENCE][table]
        if ref.empty or "t" not in ref.columns:
            continue
        row = {"analysis": table}
        for name in OTHERS:
            cur = results[name][table]
            m = ref[[*keys, "t"]].merge(cur[[*keys, "t"]], on=list(keys), suffixes=("_ref", "_cur")).dropna()
            row[name] = m["t_ref"].corr(m["t_cur"], method="spearman") if len(m) > 2 else None
        rows.append(row)
    return pd.DataFrame(rows)


def sensitive_rows(results: dict, table: str, keys: tuple[str, ...]) -> pd.DataFrame:
    cmp = ma.compare(results, table, keys)
    if cmp.empty:
        return cmp
    flags = [f"{n}_sensitive" for n in OTHERS if f"{n}_sensitive" in cmp]
    hit = cmp[cmp[flags].any(axis=1)].copy()
    hit["sensitive_in"] = hit[flags].apply(lambda r: ", ".join(f.removesuffix("_sensitive") for f in flags if r[f]),
                                           axis=1)
    sig = cmp.attrs.get("level")
    cols = [*keys, "ref_effect", *(["ref_sig"] if sig else []), "baseline_effect",
            *(["baseline_sig"] if sig else []), "sensitive_in"]
    return hit[[c for c in cols if c in hit.columns]]


def render(results: dict, extras: dict[str, pd.DataFrame], prov: dict, now: datetime) -> str:
    specs = {c: "+.4f" for c in ("ref_effect", "baseline_effect")} | {"ref_sig": ".3g", "baseline_sig": ".3g"}
    lines = [f"# Outcome-mask sensitivity audit, research period ({now:%Y-%m-%d})", "", provenance.line(prov), "",
             "Descriptive (docs/review-2026-11-10-fixes-plan.md, fix 2): every recorded research-period analysis "
             "that reads the research sample is re-run per configuration in a scratch results database. Nothing "
             "is logged in the real store; no q-value or registry state changes.", "",
             "Configurations: `baseline` (point-in-time universe, no outcome mask), `quality` (data-error jumps "
             "in the target window), `execution` (next open at the ceiling), `period` (target crosses a period "
             "boundary), `combined` (all three: the recorded definition, the reference), `combined_horizon` "
             "(all three over each target's own window). Every configuration is compared with `combined`; "
             "`combined_horizon` has the same three masks, so its difference from `combined` is due to the "
             "window alone.", "",
             "Two levels. **Statistical sensitivity** (sections 3 to 8): a test is statistically sensitive in "
             "a configuration when its effect changes sign against "
             f"`combined`, or its q (p where no q exists) crosses {ma.Q_MAX}. q-values are BH within each "
             "configuration's own scratch family (the same tests in every configuration), so they are smaller "
             "than the recorded ones, whose family is the whole hypothesis log. **Decision sensitivity** "
             "(section 3): whether a declared decision rule gives a different outcome.", "",
             "## 1. Sample per configuration", ""]
    sample = pd.concat([r["sample"].assign(config=n) for n, r in results.items()], ignore_index=True)
    sample = sample[["config", "target", "rows", "mean", "p05", "p50", "p95"]]
    lines += _md(sample, {"rows": ",", "mean": "+.4%", "p05": "+.2%", "p50": "+.2%", "p95": "+.2%"})
    lines += ["", "Share of point-in-time rows each mask removes, per year (10-session target):", ""]
    lines += _md(extras["by_year"].astype({"year": int, "rows": int}),
                 {"rows": ",", "quality": ".2%", "execution": ".2%", "period": ".2%"})
    lines += ["", "## 2. Concentration: share of each pattern's point-in-time hits removed by each mask", "",
              "The first row is all point-in-time rows. A pattern far above it is where a mask matters.", ""]
    lines += _md(extras["concentration"], {"hits": ",", "quality": ".2%", "execution": ".2%", "period": ".2%"})
    cands = ma.candidates(results)
    ref_c = cands[ma.REFERENCE]
    lines += ["", "## 3. Decision sensitivity", "",
              "Only the interaction scan has a mechanical research-stage decision rule (the candidate rule of "
              "`regime-interaction-plan.md`). The other analyses have none to re-apply here:", "",
              "- patterns: research-stage candidates were chosen at a reviewed checkpoint; their registry decisions "
              "come from validation and are not re-decided;",
              "- decile scan and event study: descriptive, no pass/fail;",
              "- cross-stock tests: decisions come from their pre-registrations and validation, not re-decided.", "",
              f"Interaction candidates (declared rule, scratch-family q): {len(ref_c)} under `combined`.", "",
              "| config | candidates | added | dropped |", "|---|---|---|---|"]
    for name, c in cands.items():
        lines.append(f"| {name} | {len(c)} | {', '.join(sorted(c - ref_c)) or '-'} | "
                     f"{', '.join(sorted(ref_c - c)) or '-'} |")
    lines += ["", "## 4. Statistical sensitivity per configuration", "",
              "Each cell: statistically sensitive tests (in brackets: material ones, i.e. significant under "
              "`combined` and changing sign).", ""]
    lines += _md(summary(results), {})
    lines += ["", "Rank agreement of the test statistics with `combined` (Spearman):", ""]
    lines += _md(ranks(results), {n: ".3f" for n in OTHERS})
    for i, (table, keys, title) in enumerate(SECTIONS, start=5):
        hit = sensitive_rows(results, table, keys)
        lines += ["", f"## {i}. {title}", ""]
        if hit.empty:
            lines.append("No statistically sensitive test.")
        else:
            lines += _md(hit, specs)
    lines += ["", "> Validated, failed and forward hypotheses are not re-decided from this audit: validation and "
              "holdout cannot be reused. A research-stage conclusion that is decision-sensitive is noted in the "
              "results document; a registry change needs `quant registry move` with a reason."]
    return "\n".join(lines) + "\n"


def extras(research: Path) -> dict[str, pd.DataFrame]:
    con = duckdb.connect()
    con.execute(f"ATTACH '{research.as_posix()}' AS rs (READ_ONLY)")
    try:
        store = _Con(con)
        return {"by_year": ma.removed_by_year(store), "concentration": ma.concentration(store),
                "prov": pd.DataFrame([provenance.collect(con, research_code_hash())])}
    finally:
        con.close()


class _Con:
    """The two helpers only need `.con` with research.duckdb attached as `rs`."""
    def __init__(self, con) -> None:
        self.con = con


def build_report(research: Path, cross: Path | None, now: datetime, log=print) -> str:
    ex = extras(research)
    prov = ex.pop("prov").iloc[0].to_dict()
    results = ma.run_audit(research, cross, now, log)
    return render(results, ex, prov, now)



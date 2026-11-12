"""Markdown report of one stored risk evaluation (rendered from the risk_* tables only, so it is reproducible)."""

from quant_research import provenance
from quant_research.backtest.metrics import WRITEDOWN_SESSIONS
from quant_research.results import ResultsStore

LIMITATIONS = (
    "A PASS means the declared policy is met with the available inputs. It does not mean the portfolio is safe or "
    "profitable.",
    "The policy is provisional: it was declared before any evaluation and not tuned on any outcome.",
    "VaR and Expected Shortfall are historical, daily, from the portfolio's own net equity curve, and need 250 "
    "(95%) or 500 (99%) finite returns. Even then the 99% tail holds about 5 returns (the tail count is in each "
    "note): they are historical estimates, not loss ceilings or stable forecasts.",
    "Beta, volatility and effective bets come from the Exposure Engine: 120 sessions of clean daily returns. A "
    "position with fewer than 80% of them makes these limits INCOMPLETE.",
    "Liquidation estimates assume selling at most the stated share of ADV20 per session with cost model v1. They do "
    "not guarantee execution. Price limits and suspensions can block a sale (see the conditional weight).",
    f"Drawdown: policy v2 checks the worst observed drawdown of the stored equity curves (history) and, "
    f"separately, the current drawdown if positions without a trade for more than {WRITEDOWN_SESSIONS} sessions were "
    f"valued at 0 (a valuation assumption, not an observed loss). Policy v1 took the worse of the two.",
    "Industries are ICB level 2 at their current classification (no history, so not point in time for a past "
    "date). Clusters are point in time: the last month end on or before the valuation date; holdings outside the "
    "cluster universe (top 200 by liquidity) have none.",
    "Group limits are computed on labelled holdings; any unlabelled holding makes them INCOMPLETE, and the "
    "Concentration section shows the unlabelled weight and a conservative value that counts it as one group.",
    "Positions must have finite, positive shares, price and value, and positions + cash must match the NAV within "
    "1%; otherwise every limit that uses the positions is INCOMPLETE.",
    "Liquidation cost uses fallback spread or volatility where the inputs are missing (cost_fallback_weight); it "
    "changes the cost estimate only, which no limit uses.",
)


def _v(x, pct: bool = False) -> str:
    if x is None:
        return "-"
    return f"{x:.2%}" if pct else f"{x:,.4g}" if abs(x) < 1e3 else f"{x:,.0f}"


PCT = {"max_position_weight", "max_industry_weight", "max_cluster_weight", "illiquid_weight", "stale_weight",
       "max_drawdown", "invested_share", "current_drawdown", "volatility_ann", "downside_volatility_ann",
       "var_95_1d", "es_95_1d", "var_99_1d", "es_99_1d", "liquidation_cost_share", "missing_returns_weight",
       "volatility_ann_covered_positions_scaled_to_nav", "conditional_weight", "max_drawdown_historical",
       "current_drawdown_write_down", "cost_fallback_weight", "max_industry_weight_labelled",
       "industry_unlabelled_weight", "max_industry_weight_unlabelled_as_one", "max_cluster_weight_labelled",
       "cluster_unlabelled_weight", "max_cluster_weight_unlabelled_as_one"}


def render(store: ResultsStore, run_id: str) -> str:
    con = store.con
    ev = con.execute("""SELECT target, valuation_date, policy_id, policy_version, model_version, status, nav, cash, notes
                        FROM risk_evaluations WHERE run_id = ?""", [run_id]).fetchone()
    if ev is None:
        raise KeyError(f"no risk evaluation {run_id}")
    target, day, pol, ver, model, status, nav, cash, notes = ev
    provisional = con.execute("SELECT config FROM risk_policies WHERE policy_id = ? AND version = ?",
                              [pol, ver]).fetchone()
    lines = [f"# Risk evaluation: {target} at {day}", "", *provenance.from_runs(con, [run_id]), "",
             f"- Run `{run_id}`; policy `{pol}/{ver}`" + (" (provisional)" if provisional and
                                                          '"provisional": true' in provisional[0] else "")
             + f"; risk model `{model}`.",
             f"- NAV {_v(nav)} VND; cash {_v(cash)} VND.", "", f"## Status: {status}", ""]
    lines += [f"- {n}" for n in (notes or "").splitlines() if n]
    lines += ["", "## Limits", "", "| limit | value | warning | breach | status | reason | affected |",
              "|---|---|---|---|---|---|---|"]
    for name, metric, value, w, b, st, reason, aff in con.execute("""
            SELECT limit_name, metric, value, warning, breach, status, reason, affected FROM risk_limit_results
            WHERE run_id = ? ORDER BY CASE status WHEN 'INCOMPLETE' THEN 0 WHEN 'BREACH' THEN 1 WHEN 'WARNING' THEN 2
            ELSE 3 END, limit_name""", [run_id]).fetchall():
        pct = metric in PCT
        lines.append(f"| {name} | {_v(value, pct)} | {_v(w, pct)} | {_v(b, pct)} | {st} | {reason or ''} | {aff or ''} |")
    rows = con.execute("SELECT scope, metric, value, note FROM risk_metrics WHERE run_id = ? ORDER BY scope, metric",
                       [run_id]).fetchall()
    for title, prefix in (("Portfolio", "portfolio"), ("Concentration", "concentration"), ("Equity curves", "curve:"),
                          ("Valuation", "valuation:"),
                          ("Liquidity", "liquidity"), ("Factor tilts (display only)", "tilt")):
        sel = [r for r in rows if r[0].startswith(prefix)]
        if sel:
            lines += ["", f"## {title}", "", "| scope | metric | value | note |", "|---|---|---|---|"]
            lines += [f"| {s} | {m} | {_v(v, m in PCT)} | {n or ''} |" for s, m, v, n in sel]
    lines += ["", "## Positions", "", "| symbol | weight | industry | cluster | beta | risk share | liquidation days | "
              "cost | stale sessions | note |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in con.execute("""SELECT symbol, weight, industry, cluster, beta, risk_share, liquidation_days, liquidation_cost,
                                   stale_sessions, note FROM risk_positions WHERE run_id = ?
                            ORDER BY weight DESC, symbol""", [run_id]).fetchall():
        lines.append(f"| {r[0]} | {_v(r[1], True)} | {r[2] or '-'} | {r[3] or '-'} | {_v(r[4])} | {_v(r[5], True)} | "
                     f"{_v(r[6])} | {_v(r[7])} | {'-' if r[8] is None else r[8]} | {r[9] or ''} |")
    lines += _reconciliation(con, run_id) + _stress(con, run_id)
    lines += ["", "## Definitions and limitations", ""] + [f"- {x}" for x in LIMITATIONS]
    return "\n".join(lines) + "\n"


def _has(con, table: str) -> bool:
    return bool(con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name = ?", [table]).fetchone()[0])


STRESS_NOTE = ("Scenarios are illustrations, not forecasts, probabilities or policy limits: each is MEASURED or "
               "INCOMPLETE, none is summed with another, and idiosyncratic return is 0. A loss is VND lost by marking "
               "positions down (negative: a gain); the liquidation cost is separate. The 61-session suspension is an assumed "
               "write-down under prolonged suspension (the valuation rule), not an expected loss or a forecast.")


def _stress(con, run_id: str) -> list[str]:
    rows = con.execute("""SELECT scenario_id, family, grp, status, valuation_loss, valuation_loss_pct, cost, cost_before,
                                 coverage, affected, reason, notes FROM risk_stress_results WHERE run_id = ?
                          ORDER BY seq""",
                       [run_id]).fetchall() if _has(con, "risk_stress_results") else []
    if not rows:
        return ["", "## Stress scenarios", "", "Not run for this evaluation (risk model v1, or --no-stress)."]
    scen = con.execute("SELECT params::JSON ->> '$.extra.scenarios' FROM research_runs WHERE run_id = ?",
                       [run_id]).fetchone()
    out = ["", "## Stress scenarios", "", f"Scenario set `{scen[0] if scen else '-'}`. {STRESS_NOTE}", "",
           "| scenario | group | status | loss | loss % NAV | cost | cost before | coverage | affected | reason / notes |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for sid, _fam, grp, st, loss, pct, cost, before, cov, aff, reason, notes in rows:
        text = "; ".join(x for x in (reason, notes) if x)
        out.append(f"| {sid} | {grp or '-'} | {st} | {_v(loss)} | {_v(pct, True)} | {_v(cost)} | {_v(before)} | "
                   f"{_v(cov, True)} | {aff or ''} | {text} |")
    return out


def _reconciliation(con, run_id: str) -> list[str]:
    if not _has(con, "risk_reconciliation"):
        return []
    rows = con.execute("""SELECT source_run, metric, stored, resimulated, rel_diff, within, note
                          FROM risk_reconciliation WHERE run_id = ?
                          ORDER BY CASE WHEN within IS NULL THEN 1 ELSE 0 END, metric""", [run_id]).fetchall()
    if not rows:
        return []
    from quant_research.risk.construction import REQUIRED
    compared = [r for r in rows if r[5] is not None]
    bad = [r[1] for r in compared if not r[5]]
    complete = {r[1] for r in compared} >= set(REQUIRED)
    head = ("reconciles with" if complete and not bad else
            f"does NOT reconcile ({len(bad)} of {len(compared)} compared metrics differ or are missing) with")
    out = ["", "## Re-simulation", "",
           f"This portfolio is a re-simulation of the declared configuration with today's code and data. It {head} "
           f"the stored run `{rows[0][0]}` (relative tolerance 1e-9, the reproducibility policy H-1). It is not the "
           "historical run; the limits and scenarios describe the re-simulation.", "",
           "| metric | stored | re-simulated | relative difference | within | note |", "|---|---|---|---|---|---|"]
    out += [f"| {m} | {_v(a)} | {_v(b)} | {'-' if d is None else f'{d:.2e}'} | "
            f"{'not compared' if w is None else 'yes' if w else 'NO'} | {note or ''} |"
            for _src, m, a, b, d, w, note in rows]
    return out

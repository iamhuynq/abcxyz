"""Markdown report of one stored risk evaluation (rendered from the risk_* tables only, so it is reproducible)."""

from quant_research import provenance
from quant_research.backtest.metrics import WRITEDOWN_SESSIONS
from quant_research.results import ResultsStore

LIMITATIONS = (
    "A PASS means the declared policy is met with the available inputs. It does not mean the portfolio is safe or "
    "profitable.",
    "The policy is provisional: it was declared before any evaluation and not tuned on any outcome.",
    "VaR and Expected Shortfall are historical, daily, from the portfolio's own net equity curve. They are not "
    "loss ceilings and need 250 (95%) or 500 (99%) returns.",
    "Beta, volatility and effective bets come from the Exposure Engine: 120 sessions of clean daily returns. A "
    "position with fewer than 80% of them makes these limits INCOMPLETE.",
    "Liquidation estimates assume selling at most the stated share of ADV20 per session with cost model v1. They do "
    "not guarantee execution. Price limits and suspensions can block a sale (see the conditional weight).",
    f"Valuation: the portfolio's own marks, and a write-down that values positions without a trade for more than "
    f"{WRITEDOWN_SESSIONS} sessions at 0. The drawdown limit is judged on the worse.",
    "Industries are ICB level 2 at their current value (no history). Clusters are point-in-time: the last month "
    "end on or before the valuation date.",
)


def _v(x, pct: bool = False) -> str:
    if x is None:
        return "-"
    return f"{x:.2%}" if pct else f"{x:,.4g}" if abs(x) < 1e3 else f"{x:,.0f}"


PCT = {"max_position_weight", "max_industry_weight", "max_cluster_weight", "illiquid_weight", "stale_weight",
       "max_drawdown", "invested_share", "current_drawdown", "volatility_ann", "downside_volatility_ann",
       "var_95_1d", "es_95_1d", "var_99_1d", "es_99_1d", "liquidation_cost_share", "missing_returns_weight",
       "volatility_ann_positions", "conditional_weight"}


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
    for title, prefix in (("Portfolio", "portfolio"), ("Equity curves", "curve:"), ("Valuation", "valuation:"),
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
    lines += ["", "## Definitions and limitations", ""] + [f"- {x}" for x in LIMITATIONS]
    return "\n".join(lines) + "\n"

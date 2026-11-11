"""Risk evaluation (docs/risk-engine-plan.md, phases B to D and F): one portfolio snapshot -> metrics, limit
results and a status, stored as a run of kind `risk` (no hypothesis is logged, so nothing enters the FDR family).

`assess` is pure: it takes the snapshot and its inputs and returns everything that is stored. Weights are
fractions of the net asset value (cash included). Beta and volatility from the Exposure Engine describe the
positions that have enough returns; they are scaled to the NAV, and any position without enough returns makes
the limits that need them INCOMPLETE (the Exposure Engine drops and renormalizes such positions).
"""

from dataclasses import asdict, dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

from quant_research import exposure
from quant_research.backtest.metrics import WRITEDOWN_SESSIONS
from quant_research.backtest.runner import PERIOD_RANGES
from quant_research.results import ResearchParams, ResultsStore
from quant_research.risk import data, limits, liquidity, metrics
from quant_research.risk.limits import LimitResult, Policy

MODEL_VERSION = "risk-v1"
ALLOWED_PERIODS = ("research", "forward")       # validation and holdout are frozen: never re-read for risk
RECONCILIATION_TOLERANCE = 0.01                  # |NAV - cash - positions| above this share of NAV: INCOMPLETE
HISTORY_ONLY = ("max_drawdown_historical",)      # limits that need only the stored equity curves, not the positions

SCHEMA = """
CREATE TABLE IF NOT EXISTS risk_policies (
    policy_id VARCHAR NOT NULL, version VARCHAR NOT NULL, config VARCHAR NOT NULL, created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (policy_id, version));
CREATE TABLE IF NOT EXISTS risk_evaluations (
    run_id VARCHAR PRIMARY KEY, target VARCHAR NOT NULL, valuation_date DATE NOT NULL, policy_id VARCHAR NOT NULL,
    policy_version VARCHAR NOT NULL, model_version VARCHAR NOT NULL, status VARCHAR NOT NULL, nav DOUBLE,
    cash DOUBLE, notes VARCHAR);
CREATE TABLE IF NOT EXISTS risk_metrics (
    run_id VARCHAR NOT NULL, scope VARCHAR NOT NULL, metric VARCHAR NOT NULL, value DOUBLE, note VARCHAR,
    PRIMARY KEY (run_id, scope, metric));
CREATE TABLE IF NOT EXISTS risk_limit_results (
    run_id VARCHAR NOT NULL, limit_name VARCHAR NOT NULL, metric VARCHAR NOT NULL, value DOUBLE, warning DOUBLE,
    breach DOUBLE, status VARCHAR NOT NULL, reason VARCHAR, affected VARCHAR, PRIMARY KEY (run_id, limit_name));
CREATE TABLE IF NOT EXISTS risk_positions (
    run_id VARCHAR NOT NULL, symbol VARCHAR NOT NULL, value DOUBLE, weight DOUBLE, industry VARCHAR,
    cluster VARCHAR, beta DOUBLE, vol_ann DOUBLE, risk_share DOUBLE, adv DOUBLE, liquidation_days DOUBLE,
    illiquid BOOLEAN, liquidation_cost DOUBLE, cost_fallback BOOLEAN, stale_sessions INTEGER, stale BOOLEAN,
    conditional BOOLEAN, note VARCHAR, PRIMARY KEY (run_id, symbol));
"""


@dataclass
class Assessment:
    status: str
    limit_results: list[LimitResult]
    metrics: list[tuple[str, str, float | None, str]]        # scope, metric, value, note
    positions: pd.DataFrame
    notes: list[str] = field(default_factory=list)


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


@dataclass(frozen=True)
class _Groups:
    top_known: float                # largest weight among labelled groups
    top_as_one: float               # the same, with every unlabelled holding counted as one more group (conservative)
    affected: tuple                 # labelled groups at or above the warning
    unlabelled: tuple               # holdings without a label
    unlabelled_weight: float


def _group_max(weights: pd.Series, labels: pd.Series, warning: float) -> _Groups:
    lacking = tuple(sorted(labels.index[labels.isna()]))
    if len(weights) == 0:
        return _Groups(0.0, 0.0, (), (), 0.0)
    known = weights[labels.notna()].groupby(labels[labels.notna()]).sum()
    top = float(known.max()) if len(known) else 0.0
    rest = float(weights[labels.isna()].sum())
    return _Groups(top, max(top, rest), tuple(sorted(known.index[known >= warning])), lacking, rest)


def _curve_metrics(name: str, equity: np.ndarray) -> tuple[list, metrics.Drawdown | None]:
    rows, dd, rets = [], metrics.drawdown(equity), metrics.returns(equity)
    scope = f"curve:{name}"
    rows += [(scope, "sessions", float(len(equity)), ""),
             (scope, "current_drawdown", dd.current if dd else None, "" if dd else "fewer than 2 valid points"),
             (scope, "max_drawdown", dd.maximum if dd else None, ""),
             (scope, "volatility_ann", metrics.volatility(rets), "daily returns, sqrt(250)"),
             (scope, "downside_volatility_ann", metrics.downside_volatility(rets), "shortfall below 0, sqrt(250)")]
    for conf in (0.95, 0.99):
        t = metrics.historical_var_es(rets, conf)
        why = t.reason
        lvl = int(round(conf * 100))
        rows += [(scope, f"var_{lvl}_1d", t.var, why or f"historical, n {t.n}, tail {t.tail}"),
                 (scope, f"es_{lvl}_1d", t.es, why or f"mean of the {t.tail} worst returns")]
    return rows, dd


@dataclass
class _Acc:
    """What the steps of `assess` collect: limit inputs and reported metrics."""
    warn: dict[str, float]
    vals: dict[str, float | None] = field(default_factory=dict)
    missing: dict[str, str] = field(default_factory=dict)
    affected: dict[str, tuple] = field(default_factory=dict)
    out: list[tuple[str, str, float | None, str]] = field(default_factory=list)


def assess(snap: data.Snapshot, inputs: pd.DataFrame, rets: dict, cluster_of: dict[str, str],
           factor_z: pd.DataFrame | None, policy: Policy,
           liq: liquidity.LiquidityParams = liquidity.LiquidityParams()) -> Assessment:
    pos = snap.positions[["symbol", "value"]].merge(inputs, on="symbol", how="left")
    nav = snap.nav
    if not nav or nav <= 0 or not np.isfinite(nav):
        res = limits.evaluate(policy, {}, missing={lim.metric: "NAV missing or not positive" for lim in policy.limits})
        return Assessment(limits.INCOMPLETE, res, [], pos, ["NAV missing or not positive"])
    issues, unvalued = _snapshot_issues(snap)
    pos["value"] = pos["value"].where(~pos["symbol"].isin(unvalued), 0.0)   # computed on, but never trusted
    pos["weight"] = pos["value"] / nav
    pos["cluster"] = pos["symbol"].map(cluster_of)
    acc = _Acc({lim.metric: lim.warning for lim in policy.limits})
    acc.out += [("portfolio", "unvalued_positions", float(len(unvalued)), ", ".join(unvalued)),
                ("portfolio", "reconciliation_gap", snap.reconciliation_gap,
                 "NAV - cash - positions" if snap.reconciliation_gap is not None else "not applicable: cash derived")]
    invested = _concentration(pos, snap, acc)
    pos = _exposure(pos, nav, invested, rets, factor_z, acc)
    pos = _liquidity(pos, nav, liq, acc)
    _drawdowns(pos, snap, acc)
    if issues:          # the positions cannot be trusted: every limit that uses them is INCOMPLETE
        for lim in policy.limits:
            if lim.metric not in HISTORY_ONLY:
                acc.missing[lim.metric] = "; ".join(issues + ([acc.missing[lim.metric]] if lim.metric in acc.missing
                                                              else []))
    results = limits.evaluate(policy, acc.vals, acc.affected, acc.missing)
    return Assessment(limits.overall(results), results, acc.out, pos, list(snap.notes))


def _snapshot_issues(snap: data.Snapshot) -> tuple[list[str], tuple[str, ...]]:
    """Why the positions cannot be trusted: a position without finite, positive shares, price and value; cash
    that is not finite; or positions + cash that miss the NAV by more than RECONCILIATION_TOLERANCE."""
    p = snap.positions
    ok = pd.Series(True, index=p.index)
    for col in ("shares", "price", "value"):
        if col in p:
            v = pd.to_numeric(p[col], errors="coerce")
            ok &= v.notna() & np.isfinite(v.fillna(np.nan).astype(float)) & (v > 0)
    unvalued = tuple(sorted(p.loc[~ok, "symbol"]))
    issues = [f"positions without a valid value: {', '.join(unvalued)}"] if unvalued else []
    if snap.cash is None or not np.isfinite(snap.cash):
        issues.append("cash missing or not finite")
    gap = snap.reconciliation_gap
    if gap is not None and (not np.isfinite(gap) or abs(gap) > RECONCILIATION_TOLERANCE * snap.nav):
        issues.append(f"positions + cash miss the NAV by {gap / snap.nav:.2%} (tolerance "
                      f"{RECONCILIATION_TOLERANCE:.0%})" if np.isfinite(gap) else "positions + cash do not reconcile")
    return issues, unvalued


def _concentration(pos: pd.DataFrame, snap: data.Snapshot, acc: _Acc) -> float:
    """Position, industry and cluster weights (fractions of the NAV); returns the invested share."""
    w = pos.set_index("symbol")["weight"]
    acc.vals["max_position_weight"] = float(w.max()) if len(w) else 0.0
    acc.affected["max_position_weight"] = tuple(sorted(w.index[w >= acc.warn.get("max_position_weight", 1.0)]))
    for metric, col in (("max_industry_weight", "industry"), ("max_cluster_weight", "cluster")):
        g = _group_max(w, pos.set_index("symbol")[col], acc.warn.get(metric, 1.0))
        acc.vals[metric], acc.affected[metric] = g.top_known, g.affected
        if g.unlabelled:
            acc.missing[metric] = (f"value covers labelled holdings only; {g.unlabelled_weight:.1%} of NAV has no "
                                   f"{col}: {', '.join(g.unlabelled)}")
        acc.out += [("concentration", f"max_{col}_weight_labelled", g.top_known, f"largest {col} among labelled holdings"),
                    ("concentration", f"{col}_unlabelled_weight", g.unlabelled_weight, ", ".join(g.unlabelled)),
                    ("concentration", f"max_{col}_weight_unlabelled_as_one", g.top_as_one,
                     f"conservative: holdings without {'an' if col[0] in 'aeiou' else 'a'} {col} counted as one "
                     "group")]
    h = metrics.hhi(pos["value"].to_numpy())
    invested = float(pos["value"].sum() / snap.nav)
    acc.out += [("portfolio", "nav", snap.nav, ""), ("portfolio", "cash", snap.cash, ""),
                ("portfolio", "invested_share", invested, ""), ("portfolio", "positions", float(len(pos)), ""),
                ("portfolio", "hhi", h, "of position values"),
                ("portfolio", "effective_positions", 1 / h if h else None, "1 / HHI")]
    return invested


def _exposure(pos: pd.DataFrame, nav: float, invested: float, rets: dict, factor_z: pd.DataFrame | None,
              acc: _Acc) -> pd.DataFrame:
    """Exposure Engine on the positions, scaled to the NAV; positions it drops make its limits INCOMPLETE."""
    exp = exposure.compute(rets["returns"], rets["market"], dict(zip(pos["symbol"], pos["value"])), factor_z=factor_z)
    covered = float(pos.loc[pos["symbol"].isin(exp.per_stock["symbol"]), "value"].sum() / nav)
    pos = pos.merge(exp.per_stock[["symbol", "beta", "vol_ann", "risk_share"]], on="symbol", how="left")
    beta, vol = _f(exp.portfolio["beta"]), _f(exp.portfolio["vol_ann"])
    acc.vals["abs_beta"] = abs(beta * covered) if beta is not None else None
    acc.vals["effective_bets"] = _f(exp.portfolio["effective_bets"])
    if exp.missing:
        acc.missing["abs_beta"] = acc.missing["effective_bets"] = (
            f"fewer than {exposure.MIN_OBS_SHARE:.0%} of {len(rets['returns'])} sessions with clean returns: "
            + ", ".join(sorted(exp.missing)))
    acc.out += [("portfolio", "beta", beta * covered if beta is not None else None,
                 f"Exposure Engine, {len(rets['returns'])} sessions, scaled by covered weight {covered:.3f}"),
                ("portfolio", "volatility_ann_covered_positions_scaled_to_nav", vol * covered if vol is not None else None,
                 "covariance of the positions with enough returns, times their NAV weight; not the realized "
                 "volatility of the equity curve (see the curve scopes)"),
                ("portfolio", "effective_bets", acc.vals["effective_bets"], "eigenvalue participation ratio"),
                ("portfolio", "missing_returns_weight", invested - covered, ", ".join(sorted(exp.missing)))]
    if exp.tilts is not None:
        acc.out += [("tilt", str(r.factor), _f(r.weighted_z), f"coverage {r.coverage:.2f}")
                    for r in exp.tilts.itertuples()]
    return pos


def _liquidity(pos: pd.DataFrame, nav: float, liq: liquidity.LiquidityParams, acc: _Acc) -> pd.DataFrame:
    rows = [liquidity.position(r.symbol, r.value, r.adv, _f(r.half_spread), _f(r.sigma),
                               None if pd.isna(r.stale_sessions) or pd.isna(r.last_trade) else int(r.stale_sessions),
                               bool(r.at_floor) if not pd.isna(r.at_floor) else False, liq)
            for r in pos.itertuples()]
    if not rows:
        acc.vals |= {"illiquid_weight": 0.0, "stale_weight": 0.0}
        return pos.assign(liquidation_days=None, illiquid=None, liquidation_cost=None, cost_fallback=None,
                          stale_sessions=None, stale=None, conditional=None, note="")
    lq = pd.DataFrame([asdict(x) for x in rows]).rename(
        columns={"days": "liquidation_days", "cost": "liquidation_cost", "reason": "note"})
    pos = pos.drop(columns=["adv", "stale_sessions"]).merge(lq.drop(columns=["value"]), on="symbol", how="left")
    for metric, col, label in (("illiquid_weight", "illiquid", "liquidity unknown (ADV20 invalid)"),
                               ("stale_weight", "stale", "no trade found")):
        unknown = tuple(sorted(pos.loc[pos[col].isna(), "symbol"]))
        flagged = pos[pos[col].fillna(False).astype(bool)]
        acc.vals[metric] = float(flagged["weight"].sum())
        acc.affected[metric] = tuple(sorted(flagged["symbol"]))
        if unknown:
            acc.missing[metric] = f"{label}: {', '.join(unknown)}"
    cost = float(pos["liquidation_cost"].dropna().sum())
    acc.out += [("liquidity", "liquidation_cost", cost, f"{liq.participation:.0%} of ADV20 per session, "
                 "fee + sell tax + cost model v1; positions with unknown ADV excluded"),
                ("liquidity", "liquidation_cost_share", cost / nav, ""),
                ("liquidity", "max_liquidation_days", _f(pos["liquidation_days"].max()), ""),
                ("liquidity", "cost_fallbacks", float(pos["cost_fallback"].fillna(False).sum()),
                 "positions whose cost used a fallback spread or volatility"),
                ("liquidity", "cost_fallback_weight", float(pos.loc[pos["cost_fallback"].fillna(False).astype(bool),
                                                                    "weight"].sum()),
                 "NAV share whose cost estimate rests on fallback inputs; it changes the cost estimate only, which no "
                 "limit uses (the illiquid limit needs a valid ADV, and a missing ADV is INCOMPLETE)"),
                ("liquidity", "conditional_weight", float(pos.loc[pos["conditional"].fillna(False).astype(bool),
                                                                  "weight"].sum()),
                 "stale, no trade found, or at the floor: assumes trading resumes")]
    return pos


def _drawdowns(pos: pd.DataFrame, snap: data.Snapshot, acc: _Acc) -> None:
    """Two different things, kept apart: the worst observed drawdown of the stored curves (history), and the
    current drawdown if stuck positions were valued at 0 (a valuation assumption). Policy v1's max_drawdown is
    the worse of the two; v2 checks each one."""
    hist: list[float] = []
    for name, (_dates, equity) in snap.curves.items():
        rows, dd = _curve_metrics(name, equity)
        acc.out += rows
        if dd:
            hist.append(dd.maximum)
    acc.vals["max_drawdown_historical"] = min(hist) if hist else None
    first = next(iter(snap.curves.values()), None)
    cur_wd = None
    if first is not None and len(first[1]) and np.all(np.isfinite(first[1])) and np.max(first[1]) > 0:
        # no trade found, or none for more than WRITEDOWN_SESSIONS: valued at 0 (valuation rule 2026-10-09)
        down = pos["stale_sessions"].isna() | (pos["stale_sessions"] > WRITEDOWN_SESSIONS) if len(pos) else []
        nav_wd = snap.nav - (float(pos.loc[down, "value"].sum()) if len(pos) else 0.0)
        cur_wd = min(nav_wd / float(np.max(first[1])) - 1, 0.0)
        acc.out += [("valuation:write-down", "nav", nav_wd, f"positions without a trade for more than "
                     f"{WRITEDOWN_SESSIONS} sessions valued at 0"),
                    ("valuation:write-down", "current_drawdown", cur_wd, "against the peak of the first curve")]
    acc.vals["current_drawdown_write_down"] = cur_wd
    both = [x for x in (acc.vals["max_drawdown_historical"], cur_wd) if x is not None]
    acc.vals["max_drawdown"] = min(both) if both else None
    if not hist:
        acc.missing["max_drawdown_historical"] = acc.missing["max_drawdown"] = "no equity curve with 2 valid points"
    if cur_wd is None:
        acc.missing["current_drawdown_write_down"] = "no equity curve with a positive peak"


def period_of(day: date) -> str:
    for name, (lo, hi) in PERIOD_RANGES.items():
        if lo <= day <= hi:
            return name
    raise ValueError(f"date {day} outside every period")


def register_policy(store: ResultsStore, policy: Policy, now: datetime) -> None:
    """Policies are frozen: the same id and version with another configuration is refused."""
    row = store.con.execute("SELECT config FROM risk_policies WHERE policy_id = ? AND version = ?",
                            [policy.policy_id, policy.version]).fetchone()
    if row is None:
        store.con.execute("INSERT INTO risk_policies VALUES (?, ?, ?, ?)",
                          [policy.policy_id, policy.version, policy.to_json(), now])
    elif row[0] != policy.to_json():
        raise ValueError(f"policy {policy.key} is stored with another configuration; create a new version")


def evaluate(store: ResultsStore, snap: data.Snapshot, policy: Policy, now: datetime,
             liq: liquidity.LiquidityParams = liquidity.LiquidityParams()) -> tuple[str, Assessment]:
    period = period_of(snap.valuation_date)
    if period not in ALLOWED_PERIODS:
        raise ValueError(f"valuation date {snap.valuation_date} is in the {period} period, which is frozen; "
                         f"risk evaluations use {' or '.join(ALLOWED_PERIODS)} dates only")
    store.con.execute(SCHEMA)
    register_policy(store, policy, now)
    slug = snap.target.replace(":", "-").replace("/", "-")
    run_id = f"{now:%Y%m%dT%H%M%S}-risk-{slug}-{period}"
    params = ResearchParams(extra={"target": snap.target, "valuation_date": str(snap.valuation_date),
                                   "policy": policy.key, "model_version": MODEL_VERSION, "liquidity": asdict(liq),
                                   "exposure_sessions": data.EXPOSURE_SESSIONS})
    store.start_run(run_id, "risk", period, params, now)
    symbols = tuple(snap.positions["symbol"])
    a = assess(snap, data.market_inputs(store.con, symbols, snap.valuation_date),
               data.exposure_returns(store.con, symbols, snap.valuation_date, schema="rs.") if symbols else
               {"returns": pd.DataFrame(), "market": pd.Series(dtype=float)},
               data.clusters(store.con, symbols, snap.valuation_date), data.factor_z(store.con, snap.valuation_date),
               policy, liq)
    _store(store, run_id, snap, policy, a)
    store.finish_run(run_id)
    return run_id, a


def _store(store: ResultsStore, run_id: str, snap: data.Snapshot, policy: Policy, a: Assessment) -> None:
    con = store.con
    con.execute("INSERT INTO risk_evaluations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [run_id, snap.target, snap.valuation_date, policy.policy_id, policy.version, MODEL_VERSION, a.status,
                 snap.nav, snap.cash, "\n".join(a.notes)])
    con.executemany("INSERT INTO risk_metrics VALUES (?, ?, ?, ?, ?)",
                    [(run_id, s, m, _f(v), n) for s, m, v, n in a.metrics])
    con.executemany("INSERT INTO risk_limit_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(run_id, r.limit, r.metric, _f(r.value), r.warning, r.breach, r.status, r.reason,
                      ", ".join(r.affected)) for r in a.limit_results])
    cols = ["symbol", "value", "weight", "industry", "cluster", "beta", "vol_ann", "risk_share", "adv", "liquidation_days",
            "illiquid", "liquidation_cost", "cost_fallback", "stale_sessions", "stale", "conditional", "note"]
    p = a.positions.reindex(columns=cols)
    con.executemany(f"INSERT INTO risk_positions VALUES (?, {', '.join(['?'] * len(cols))})",
                    [(run_id, *[None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in row])
                     for row in p.itertuples(index=False)])

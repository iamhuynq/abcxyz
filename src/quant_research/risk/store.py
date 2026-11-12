"""Persistence of risk evaluations (additive tables in results.duckdb; nothing existing is altered).

Policies and scenario sets are frozen: the same id and version with another configuration is refused.
"""

from datetime import datetime

import numpy as np
import pandas as pd

from quant_research.results import ResultsStore
from quant_research.risk import data
from quant_research.risk.limits import Policy
from quant_research.risk.stress import ScenarioSet, StressResult

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
CREATE TABLE IF NOT EXISTS risk_scenarios (
    set_id VARCHAR NOT NULL, version VARCHAR NOT NULL, config VARCHAR NOT NULL, created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (set_id, version));
CREATE TABLE IF NOT EXISTS risk_stress_results (
    run_id VARCHAR NOT NULL, seq INTEGER NOT NULL, scenario_id VARCHAR NOT NULL, family VARCHAR NOT NULL,
    grp VARCHAR NOT NULL, status VARCHAR NOT NULL, nav_before DOUBLE, valuation_loss DOUBLE, valuation_loss_pct DOUBLE, cost DOUBLE,
    cost_before DOUBLE, coverage DOUBLE, affected VARCHAR, reason VARCHAR, notes VARCHAR,
    PRIMARY KEY (run_id, scenario_id, grp));
CREATE TABLE IF NOT EXISTS risk_reconciliation (
    run_id VARCHAR NOT NULL, source_run VARCHAR NOT NULL, metric VARCHAR NOT NULL, stored DOUBLE,
    resimulated DOUBLE, rel_diff DOUBLE, within BOOLEAN, PRIMARY KEY (run_id, metric));
"""


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)


def _frozen(store: ResultsStore, table: str, keys: tuple[str, str], config: str, now: datetime, what: str) -> None:
    row = store.con.execute(f"SELECT config FROM {table} WHERE {keys[0]} = ? AND {keys[1]} = ?", list(what.split("/"))
                            ).fetchone()
    if row is None:
        store.con.execute(f"INSERT INTO {table} VALUES (?, ?, ?, ?)", [*what.split("/"), config, now])
    elif row[0] != config:
        raise ValueError(f"{what} is stored with another configuration; create a new version")


def register_policy(store: ResultsStore, policy: Policy, now: datetime) -> None:
    _frozen(store, "risk_policies", ("policy_id", "version"), policy.to_json(), now, policy.key)


def register_scenarios(store: ResultsStore, scen: ScenarioSet, now: datetime) -> None:
    _frozen(store, "risk_scenarios", ("set_id", "version"), scen.to_json(), now, scen.key)


def save_assessment(store: ResultsStore, run_id: str, snap: data.Snapshot, policy: Policy, a, model_version: str) -> None:
    con = store.con
    con.execute("INSERT INTO risk_evaluations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [run_id, snap.target, snap.valuation_date, policy.policy_id, policy.version, model_version, a.status,
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


def save_stress(store: ResultsStore, run_id: str, results: list[StressResult]) -> None:
    store.con.executemany("INSERT INTO risk_stress_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                          [(run_id, seq, r.scenario_id, r.family, r.group, r.status, _f(r.nav_before), _f(r.valuation_loss),
                            _f(r.valuation_loss_pct), _f(r.cost), _f(r.cost_before), _f(r.coverage),
                            ", ".join(r.affected), r.reason, r.notes) for seq, r in enumerate(results)])


def save_reconciliation(store: ResultsStore, run_id: str, source_run: str, rows: list[tuple]) -> None:
    """rows: (metric, stored, resimulated, rel_diff, within)."""
    store.con.executemany("INSERT INTO risk_reconciliation VALUES (?, ?, ?, ?, ?, ?, ?)",
                          [(run_id, source_run, m, _f(a), _f(b), _f(d), bool(w)) for m, a, b, d, w in rows])

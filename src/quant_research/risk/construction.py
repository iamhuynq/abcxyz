"""Construction-grid configurations for the Risk Engine (docs/risk-engine-pr2-plan.md, decision D1).

The historical grid runs cannot be reproduced: their warehouse file is not kept and their code was uncommitted
(Phase 0 audit). So a construction portfolio is a **current re-simulation** of the declared configuration, with
the configuration stored by the latest valid run (its version must match), at the decision setting (1 bn VND,
cost model v1), with today's code and data. It is never presented as the historical run: the target is
`construction:<name>@resim`, and a reconciliation table compares it with that run's stored metrics (relative
tolerance 1e-9, the reproducibility policy H-1). Nothing is written to the grid's tables.
"""

import json
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from quant_research.backtest.data import UniverseRule
from quant_research.backtest.metrics import WRITEDOWN_SESSIONS, participation
from quant_research.portfolio import evaluate as grid
from quant_research.portfolio.construction import Construction
from quant_research.results import ResultsStore
from quant_research.risk.data import Snapshot

TOLERANCE = 1e-9
METHOD = "current re-simulation (today's code and data, decision setting 1 bn VND, cost model v1 k = 1)"


@dataclass(frozen=True)
class Source:
    run_id: str
    name: str
    config: grid.Config
    rule: UniverseRule
    version: str


def source(store: ResultsStore, name: str) -> Source:
    """The latest valid run of the configuration, and the configuration and universe it stored."""
    row = store.con.execute("""SELECT run_id, pattern_version, params FROM research_runs WHERE kind = 'portfolio'
                               AND status = 'ok' AND pattern_name = ? ORDER BY created_at DESC LIMIT 1""",
                            [name]).fetchone()
    if row is None:
        raise ValueError(f"no valid construction run of configuration {name!r}")
    run_id, version, params = row
    extra = json.loads(params)["extra"]
    c = extra["config"]
    cfg = grid.Config(c["signal"], c["schedule"], Construction(**c["construction"]),
                      tuple(c["cash_when"]) if c.get("cash_when") else None)
    if grid.config_version(cfg) != version:
        raise ValueError(f"run {run_id} stored configuration version {version}, but its stored parameters give "
                         f"{grid.config_version(cfg)}: refusing to re-simulate something else")
    return Source(run_id, name, cfg, UniverseRule(**extra["universe"]), version)


def _positions(market, res, i: int) -> pd.DataFrame:
    """Shares held at the close of session i (from the trades) at their last traded close (the engine's mark)."""
    shares = np.zeros(market.open.shape[1])
    for t, j, q, _price, _cost in res.trades:
        if t <= i:
            shares[j] += q
    held = np.flatnonzero(shares > 1e-9)
    rows = []
    for j in held:
        traded = np.flatnonzero(market.traded[:i + 1, j] & np.isfinite(market.close[:i + 1, j]))
        price = float(market.close[traded[-1], j]) if len(traded) else np.nan
        rows.append((str(market.symbols[j]), float(shares[j]), price, float(shares[j]) * price))
    return pd.DataFrame(rows, columns=["symbol", "shares", "price", "value"])


def reconciliation(store: ResultsStore, src: Source, res, market) -> list[tuple]:
    """(metric, stored, re-simulated, relative difference, within) for every stored strategy metric of the decision
    setting that this simulation recomputes."""
    now = grid._metrics(res, grid.DECISION[0])
    now |= participation(np.array([abs(t[2]) * t[3] for t in res.trades]),
                         np.array([market.adv_value[t[0] - 1, t[1]] for t in res.trades]))
    stored = dict(store.con.execute("""SELECT metric, value FROM portfolio_results WHERE run_id = ? AND capital = ?
                                       AND cost_model = ? AND series = 'strategy'""",
                                    [src.run_id, grid.DECISION[0], grid.DECISION[1]]).fetchall())
    out = []
    for metric in sorted(set(stored) & set(now)):
        a, b = stored[metric], now[metric]
        ok_a, ok_b = a is not None and np.isfinite(a), b is not None and np.isfinite(b)
        if not ok_a or not ok_b:
            out.append((metric, a, b, None, ok_a == ok_b))
            continue
        diff = abs(b - a) / abs(a) if a else abs(b - a)
        out.append((metric, a, b, diff, diff <= TOLERANCE))
    return out


def snapshot(store: ResultsStore, name: str, on: date | None = None) -> tuple[Snapshot, Source, list[tuple]]:
    src = source(store, name)
    prep = grid.prepare(store, src.config, "research", src.rule)
    res = grid.simulate(prep, src.config)
    written = grid.simulate(prep, src.config, writedown_after=WRITEDOWN_SESSIONS)
    dates = prep.market.dates.astype("datetime64[D]").astype(object)
    i = len(dates) - 1 if on is None else int(np.searchsorted(dates, on, side="right")) - 1
    if i < 0:
        raise ValueError(f"{on} is before the first session of configuration {name}")
    pos = _positions(prep.market, res, i)
    nav, cash = float(res.equity[i]), float(res.cash[i])
    gap = nav - cash - float(pos["value"].sum())
    gap = 0.0 if abs(gap) < 0.5 else gap
    rec = reconciliation(store, src, res, prep.market)
    bad = [m for m, *_rest, w in rec if not w]
    notes = [f"Re-simulation of {name} (version {src.version}) with today's code and data; NOT the historical run "
             f"{src.run_id}. It " + ("reconciles with it." if not bad else
                                     f"does not reconcile with it: {', '.join(bad)} differ beyond {TOLERANCE:g}."),
             "Positions at the engine's marks (last traded close); a second curve values positions without a trade "
             f"for more than {WRITEDOWN_SESSIONS} sessions at 0."]
    curves = {"default marks (last traded close)": (dates[:i + 1], res.equity[:i + 1]),
              f"write-down {WRITEDOWN_SESSIONS}": (dates[:i + 1], written.equity[:i + 1])}
    snap = Snapshot(f"construction:{name}@resim", dates[i], nav, cash, pos, curves, notes, gap)
    return snap, src, rec

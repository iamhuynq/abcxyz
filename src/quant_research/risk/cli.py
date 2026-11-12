"""`quant risk`: evaluate a portfolio snapshot against a versioned risk policy (descriptive; no hypothesis logged)."""

import argparse
import sys
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from quant_research.results import ResultsStore
from quant_research.risk import construction, data, evaluate, limits, report, stress

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def add_parser(sub: argparse._SubParsersAction) -> None:
    rk = sub.add_parser("risk", help="Risk Engine: limits, liquidity and drawdown of a portfolio snapshot "
                                     "(descriptive, never changes a portfolio)")
    rs = rk.add_subparsers(dest="risk_command", required=True)
    ev = rs.add_parser("evaluate", help="Evaluate the current paper portfolio or a stored backtest run")
    target = ev.add_mutually_exclusive_group(required=True)
    target.add_argument("--paper", action="store_true", help="the current paper-portfolio record")
    target.add_argument("--backtest", metavar="RUN", help="a stored backtest run id")
    target.add_argument("--construction", metavar="CONFIG",
                        help="a construction-grid configuration (C1..C6), re-simulated now: never the historical run")
    ev.add_argument("--date", help="valuation date for --backtest / --construction (YYYY-MM-DD; default: the last "
                                   "session)")
    ev.add_argument("--policy", default=limits.DEFAULT.key, choices=sorted(limits.POLICIES))
    ev.add_argument("--no-stress", action="store_true", help="skip the stress scenarios (stress/v1)")
    rs.add_parser("policies", help="List the declared risk policies and their limits")
    rp = rs.add_parser("report", help="Re-render the report of a stored risk run")
    rp.add_argument("run_id")


def run(args: argparse.Namespace, store_factory: Callable[[], ResultsStore],
        write_report: Callable[[str, str], Path], cross_path: Path) -> int:
    if args.risk_command == "policies":
        for policy in limits.POLICIES.values():
            print(f"{policy.key}{' (provisional)' if policy.provisional else ''}: {policy.note}")
            for lim in policy.limits:
                print(f"  {lim.name}: {lim.metric} {lim.direction} warning {lim.warning:g}, breach {lim.breach:g}"
                      f" - {lim.rationale}")
        return 0
    store = store_factory()
    try:
        if args.risk_command == "report":
            text = report.render(store, args.run_id)
            print(f"Risk report: {write_report(f'risk-{args.run_id}', text)}")
            return 0
        if cross_path.exists():
            store.con.execute(f"ATTACH '{cross_path.as_posix()}' AS cx (READ_ONLY)")
        extra, rec = None, None
        if args.paper:
            snap = data.paper_snapshot(store.con)
            missing = "no current paper-portfolio record"
        elif args.construction:
            on = date.fromisoformat(args.date) if args.date else None
            try:
                snap, src, rec = construction.snapshot(store, args.construction, on)
            except ValueError as exc:
                print(f"Refused: {exc}", file=sys.stderr)
                return 1
            extra = {"source_run": src.run_id, "config_version": src.version, "method": construction.METHOD,
                     "reconciles": construction.reconciles(rec)}
            missing = ""
        else:
            on = date.fromisoformat(args.date) if args.date else None
            snap = data.backtest_snapshot(store.con, args.backtest, on)
            missing = f"no stored daily equity for backtest run {args.backtest}"
        if snap is None:
            print(f"Refused: {missing}.", file=sys.stderr)
            return 1
        now = datetime.now(VN_TZ).replace(microsecond=0)
        try:
            run_id, a = evaluate.evaluate(store, snap, limits.POLICIES[args.policy], now,
                                          scenarios=None if args.no_stress else stress.STRESS_V1, extra=extra,
                                          reconciliation=(extra["source_run"], rec) if rec is not None else None)
        except ValueError as exc:
            print(f"Refused: {exc}", file=sys.stderr)
            return 1
        if rec is not None:
            print(snap.notes[0])
        print(f"{snap.target} at {snap.valuation_date}: {a.status}")
        for r in a.limit_results:
            if r.status != limits.PASS:
                print(f"  {r.status:<10} {r.limit}: {r.reason}" + (f" [{', '.join(r.affected)}]" if r.affected else ""))
        print(f"Risk report: {write_report(f'risk-{run_id}', report.render(store, run_id))}")
        return 0
    finally:
        store.close()

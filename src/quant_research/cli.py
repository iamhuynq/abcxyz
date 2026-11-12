"""`uv run quant <command>`: build, status, sanity, export for the research database."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb

from fireant_crawler.config import load_settings
from fireant_crawler.dotenv import read_env_file
from quant_research.backtest.engine import Strategy
from quant_research.backtest.report import GRID, grid_label, render_backtest
from quant_research.backtest.runner import run_backtest
from quant_research import registry_cli
from quant_research.ml import cli as ml_cli
from quant_research.cross import cli as cross_cli
from quant_research.risk import cli as risk_cli
from quant_research.daily import run_daily
from quant_research.event_study import render_event_study, run_event_study
from quant_research.daily_report import write_reports
from quant_research.build import FEATURE_SET_VERSION, BuildError, build, feature_build_hash, sql_files
from quant_research.engine import run_pattern, run_scan
from quant_research.patterns import LIBRARY, Pattern
from quant_research.report import render_pattern_batch, render_scan
from quant_research.results import HoldoutLockedError, ResearchParams, ResultsStore
from quant_research.sanity import render_sanity_report

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
EXPORT_TABLES = ("daily_panel", "market_daily", "stock_features", "stock_targets")


def _paths() -> tuple[Path, Path]:
    settings = load_settings({**read_env_file(Path(os.environ.get("ENV_FILE", ".env"))), **os.environ})
    return settings.data_dir / "warehouse.duckdb", settings.data_dir / "research.duckdb"


def cmd_build(dry_run: bool) -> int:
    warehouse, research = _paths()
    if dry_run:
        print(f"Dry run. Would rebuild {research} from {warehouse} (READ_ONLY).")
        print(f"Feature set {FEATURE_SET_VERSION}, feature-build hash {feature_build_hash()}; SQL steps:")
        for name, _ in sql_files():
            print(f"  {name}")
        return 0
    now = datetime.now(VN_TZ).replace(microsecond=0)
    try:
        result = build(warehouse, research, now)
    except BuildError as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
        return 4
    print(f"Build {result.build_id} done in {result.seconds}s; data as of {result.data_as_of}; code {result.code_hash}")
    for table, rows in result.rows.items():
        print(f"  {table}: {rows:,}")
    print("Warehouse file hash unchanged.")
    return 0


def _open_research() -> duckdb.DuckDBPyConnection | None:
    _, research = _paths()
    if not research.exists():
        print(f"No research database at {research}; run `quant build`", file=sys.stderr)
        return None
    return duckdb.connect(str(research), read_only=True)


def cmd_status() -> int:
    con = _open_research()
    if con is None:
        return 1
    with con:
        for row in con.execute(
            "SELECT build_id, feature_set_version, data_as_of, code_hash, feature_rows, seconds "
            "FROM feature_builds ORDER BY built_at DESC LIMIT 5"
        ).fetchall():
            print(f"  {row}")
    current = feature_build_hash()
    print(f"Current code hash: {current}")
    return 0


def cmd_sanity() -> int:
    con = _open_research()
    if con is None:
        return 1
    now = datetime.now(VN_TZ).replace(microsecond=0)
    with con:
        report = render_sanity_report(con, now)
    warehouse, _ = _paths()
    out = warehouse.parent / "reports" / f"feature-sanity-{now.date().isoformat()}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Report: {out}")
    return 0


def cmd_export() -> int:
    con = _open_research()
    if con is None:
        return 1
    warehouse, _ = _paths()
    out_dir = warehouse.parent / "parquet" / "research"
    out_dir.mkdir(parents=True, exist_ok=True)
    with con:
        for table in EXPORT_TABLES:
            target = out_dir / f"{table}.parquet"
            tmp = target.with_suffix(".parquet.tmp")
            con.execute(f"COPY {table} TO '{tmp.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
            tmp.replace(target)
            print(f"  {target}")
    return 0


def _store() -> ResultsStore:
    warehouse, research = _paths()
    return ResultsStore(warehouse.parent / "results.duckdb", research)


def _write_report(name: str, text: str) -> Path:
    warehouse, _ = _paths()
    out = warehouse.parent / "reports" / "research" / f"{name}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def _prereg_extra(period: str, prereg: Path | None) -> dict | None:
    """Validation and holdout runs must cite a pre-registration file; its SHA-256 is stored with the run."""
    if period == "research":
        return {}
    if prereg is None or not prereg.is_file():
        print(f"The {period} period requires --prereg <file> written before the run.", file=sys.stderr)
        return None
    import hashlib
    return {"preregistration": str(prereg), "sha256": hashlib.sha256(prereg.read_bytes()).hexdigest()}


def cmd_pattern(names: list[str], where: str | None, period: str, final: bool, prereg: Path | None = None) -> int:
    now = datetime.now(VN_TZ).replace(microsecond=0)
    extra = _prereg_extra(period, prereg)
    if extra is None:
        return 2
    if where:
        patterns = [Pattern(names[0] if names else "custom", where, "user-defined condition")]
    else:
        unknown = [n for n in names if n not in LIBRARY]
        if unknown:
            print(f"Unknown patterns {unknown}; available: {', '.join(LIBRARY)}", file=sys.stderr)
            return 2
        patterns = [LIBRARY[n] for n in (names or LIBRARY)]
    params = ResearchParams(extra=extra)
    with _store() as store:
        run_ids = []
        for pattern in patterns:
            run_ids.append(run_pattern(store, pattern, period, params, now, final=final))
            print(f"  {pattern.name}: {run_ids[-1]}")
        report = render_pattern_batch(store.con, run_ids, period, now)
    print(f"Report: {_write_report(f'patterns-{period}-{now:%Y%m%dT%H%M%S}', report)}")
    return 0


def cmd_scan(period: str, final: bool, prereg: Path | None = None) -> int:
    now = datetime.now(VN_TZ).replace(microsecond=0)
    extra = _prereg_extra(period, prereg)
    if extra is None:
        return 2
    with _store() as store:
        run_id = run_scan(store, period, ResearchParams(extra=extra), now, final=final)
        report = render_scan(store.con, run_id, now)
    print(f"Run {run_id}. Report: {_write_report(f'scan-{period}-{now:%Y%m%dT%H%M%S}', report)}")
    return 0


def cmd_backtest(grid: bool, hold: int, positions: int, renew: bool, period: str, final: bool,
                 prereg: Path | None, equity: float = 10e9) -> int:
    now = datetime.now(VN_TZ).replace(microsecond=0)
    extra = _prereg_extra(period, prereg)
    if extra is None:
        return 2
    configs = GRID if grid else [(hold, positions, renew)]
    with _store() as store:
        run_ids = []
        for h, k, r in configs:
            label = grid_label(h, k, r) + ("" if equity == 10e9 else f"_eq{equity / 1e9:g}bn")
            run_ids.append(run_backtest(store, label, Strategy(hold_sessions=h, max_positions=k, renew=r,
                                                               initial_equity=equity),
                                        period, now, final=final, extra=extra))
            print(f"  {label}: {run_ids[-1]}")
        report = render_backtest(store.con, run_ids, now)
    warehouse, _ = _paths()
    out = warehouse.parent / "reports" / "backtest" / f"backtest-{period}-{now:%Y%m%dT%H%M%S}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(f"Report: {out}")
    return 0


EXIT_INCOMPLETE = 7


def cmd_daily(rescan: str | None) -> int:
    from datetime import date as _date
    now = datetime.now(VN_TZ).replace(microsecond=0)
    with _store() as store:
        try:
            result = run_daily(store, now, rescan=_date.fromisoformat(rescan) if rescan else None)
        except ValueError as exc:                     # bad date, or a session that is not final yet
            print(f"Refused: {exc}", file=sys.stderr)
            return 2
        warehouse, _ = _paths()
        reports = write_reports(store, result, now, warehouse.parent / "reports" / "daily")
    print(f"Latest data {result.latest_date}; scanned {len(result.scanned)} session(s); "
          f"incomplete {len(result.incomplete)}; paper sessions {result.paper_sessions}.")
    print(f"Reports: {', '.join(p.name for p in reports)} in {reports[-1].parent}")
    latest_incomplete = any(d == result.latest_date for d, _ in result.incomplete)
    return EXIT_INCOMPLETE if latest_incomplete else 0


def cmd_event_study() -> int:
    now = datetime.now(VN_TZ).replace(microsecond=0)
    with _store() as store:
        run_id = run_event_study(store, now)
        text = render_event_study(store, run_id, ResearchParams().cost_round_trip)
    print(f"Event study {run_id} (descriptive, not logged). Report: {_write_report(f'events-{run_id}', text)}")
    return 0


def cmd_interactions(period: str) -> int:
    from quant_research.interaction_report import render, tests_with_q
    from quant_research.interactions import InteractionRefused, run_scan
    now = datetime.now(VN_TZ).replace(microsecond=0)
    with _store() as store:
        try:
            run_id = run_scan(store, period, now)
        except InteractionRefused as exc:
            print(f"Refused: {exc}", file=sys.stderr)
            return 2
        text = render(store, run_id)
        n = int(tests_with_q(store, run_id)["candidate"].sum())
    print(f"Interaction scan {run_id}: {n} candidate(s). Report: {_write_report(f'interactions-{run_id}', text)}")
    return 0


def cmd_econ(strategy_name: str, period: str) -> int:
    from quant_research.daily import FROZEN_STRATEGY
    from quant_research.econ import EconRefused, evaluate
    from quant_research.econ_report import render
    strategies = {"frozen": FROZEN_STRATEGY}          # the paper portfolio's strategy (version 72c851c7)
    now = datetime.now(VN_TZ).replace(microsecond=0)
    with _store() as store:
        try:
            run_id = evaluate(store, strategy_name, strategies[strategy_name], period, now)
        except EconRefused as exc:
            print(f"Refused: {exc}", file=sys.stderr)
            return 2
        text = render(store, run_id)
    print(f"Economic evaluation {run_id} (descriptive, not logged). Report: {_write_report(f'econ-{run_id}', text)}")
    return 0


def cmd_portfolio(config: str, period: str) -> int:
    from quant_research.portfolio.evaluate import CONFIGS, PortfolioRefused, evaluate
    from quant_research.portfolio.report import latest_runs, render
    names = list(CONFIGS) if config == "all" else [config]
    now = datetime.now(VN_TZ).replace(microsecond=0)
    with _store() as store:
        for name in names:
            try:
                run_id = evaluate(store, name, now, period)
            except PortfolioRefused as exc:
                print(f"Refused ({name}): {exc}", file=sys.stderr)
                return 2
            print(f"{name}: {run_id}")
        text = render(store, {k: v for k, v in latest_runs(store).items() if k in CONFIGS})
    print(f"Report: {_write_report(f'portfolio-grid-{now:%Y%m%dT%H%M%S}', text)}")
    return 0


def cmd_audit_masks() -> int:
    from quant_research.mask_audit_report import build_report
    _, research = _paths()
    now = datetime.now(VN_TZ).replace(microsecond=0)
    cross = research.parent / "cross.duckdb"
    text = build_report(research, cross if cross.exists() else None, now)
    print(f"Outcome-mask audit (descriptive, not logged): {_write_report(f'audit-masks-{now:%Y%m%dT%H%M%S}', text)}")
    return 0


def cmd_audit_fdr() -> int:
    from quant_research.fdr_audit import render
    now = datetime.now(VN_TZ).replace(microsecond=0)
    store = _store()
    try:
        text = render(store, now)
    finally:
        store.close()
    print(f"Multiple-testing audit (descriptive, not logged): {_write_report(f'audit-fdr-{now:%Y%m%dT%H%M%S}', text)}")
    return 0


def cmd_audit() -> int:
    from quant_research.audit import render
    warehouse, research = _paths()
    build_id, text = render(warehouse, research)
    print(f"Point-in-time audit (descriptive, not logged): {_write_report(f'audit-pit-{build_id}', text)}")
    return 0


def cmd_factors_describe() -> int:
    from quant_research.factor_report import render
    _, research = _paths()
    try:
        build_id, text = render(research)
    except RuntimeError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 1
    print(f"Factor structure (descriptive, research period, not logged): {_write_report(f'factors-{build_id}', text)}")
    return 0


def cmd_runs(invalidate: str | None, reason: str | None) -> int:
    with _store() as store:
        if invalidate:
            if not reason:
                print("--reason is required to invalidate a run", file=sys.stderr)
                return 2
            if not store.invalidate(invalidate, reason):
                row = store.con.execute("SELECT status, invalid_reason FROM research_runs WHERE run_id = ?",
                                        [invalidate]).fetchone()
                if row and row[0] == "invalid":
                    print(f"Run {invalidate} is already invalid; its reason is kept: {row[1]}", file=sys.stderr)
                else:
                    print(f"No run {invalidate}; nothing changed.", file=sys.stderr)
                return 1
            print(f"Run {invalidate} marked invalid (kept in the log, excluded from q-values).")
            return 0
        for row in store.con.execute("""SELECT run_id, kind, pattern_name, period, status, feature_build_id
                                        FROM research_runs ORDER BY created_at DESC LIMIT 30""").fetchall():
            print(f"  {row}")
        print(f"Holdout runs so far: {store.holdout_runs()}")
        stuck = [r for r in store.stuck_runs() if r[0] not in store._open_runs]
        if stuck:
            print(f"Runs left 'running' by a killed process ({len(stuck)}); they count nowhere. Invalidate each with "
                  "`quant runs --invalidate RUN --reason \"process killed\"` and run it again:")
            for run_id, created in stuck:
                print(f"  {run_id} (started {created})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quant", description="Research layer: features and targets")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="Rebuild data/research.duckdb from the warehouse (read-only)")
    b.add_argument("--dry-run", action="store_true")
    sub.add_parser("status", help="Show recent builds")
    sub.add_parser("sanity", help="Write a sanity report for the latest build")
    sub.add_parser("export", help="Export research tables to data/parquet/research")
    for name, help_text in (("pattern", "Test patterns (default: the whole library)"),
                            ("scan", "Decile scan of every feature x horizon")):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--period", default="research", choices=("research", "validation", "holdout", "forward"))
        sp.add_argument("--final", action="store_true", help="Required for the holdout period (logged)")
        sp.add_argument("--prereg", type=Path, help="Pre-registration file (required outside the research period)")
        if name == "pattern":
            sp.add_argument("--name", action="append", default=[], help="Library pattern name (repeatable)")
            sp.add_argument("--where", help="Ad-hoc SQL condition on feature_target columns (with --name)")
    bt = sub.add_parser("backtest", help="Portfolio backtest of the order-imbalance signal")
    bt.add_argument("--grid", action="store_true", help="Run the 12 pre-declared variants")
    bt.add_argument("--hold", type=int, default=10)
    bt.add_argument("--positions", type=int, default=20)
    bt.add_argument("--no-renew", action="store_true")
    bt.add_argument("--period", default="research", choices=("research", "validation", "holdout", "forward"))
    bt.add_argument("--final", action="store_true")
    bt.add_argument("--prereg", type=Path)
    bt.add_argument("--equity", type=float, default=10e9, help="Initial equity in VND (default 10e9)")
    dl = sub.add_parser("daily", help="Scan new sessions, update the paper portfolio, write the daily report")
    dl.add_argument("--date", help="Re-scan one session (YYYY-MM-DD)")
    runs = sub.add_parser("runs", help="List runs, or invalidate one (kept in the log)")
    runs.add_argument("--invalidate", help="Run id. Only for a defect found independently of the result (code, data "
                                           "or method error), never because a result is unwelcome")
    runs.add_argument("--reason", help="The defect, named (required with --invalidate)")
    ev = sub.add_parser("events", help="Event catalog (Phase 5)")
    ev.add_argument("events_command", choices=("study",), help="study: descriptive event study, research period")
    pf = sub.add_parser("portfolio", help="Portfolio construction grid (research period, one logged test per config)")
    pf.add_argument("portfolio_command", choices=("evaluate",))
    pf.add_argument("--config", default="all", choices=("all", "C1", "C2", "C3", "C4", "C5", "C6"))
    pf.add_argument("--period", default="research", choices=("research", "validation", "holdout", "forward"))
    ec = sub.add_parser("econ", help="Economic evaluation: cost models, capital, benchmarks, matched controls")
    ec.add_argument("econ_command", choices=("evaluate",))
    ec.add_argument("--strategy", default="frozen", choices=("frozen",))
    ec.add_argument("--period", default="research", choices=("research", "validation", "holdout", "forward"))
    ix = sub.add_parser("interactions", help="Interaction Engine: factor IC by market regime (logged)")
    ix.add_argument("interactions_command", choices=("scan",))
    ix.add_argument("--period", default="research", choices=("research", "validation", "holdout", "forward"))
    au = sub.add_parser("audit", help="Data audits (descriptive, not logged)")
    au.add_argument("audit_command", choices=("pit", "masks", "fdr"),
                    help="pit: point-in-time and survivorship audit; masks: outcome-mask sensitivity (research); "
                         "fdr: BY q-values next to BH")
    fc = sub.add_parser("factors", help="Factor Engine (factor set f1)")
    fc.add_argument("factors_command", choices=("describe",),
                    help="describe: factor structure on the research period (descriptive, no returns)")
    cross_cli.add_parser(sub)
    registry_cli.add_parser(sub)
    ml_cli.add_parser(sub)
    risk_cli.add_parser(sub)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            return cmd_build(args.dry_run)
        if args.command == "pattern":
            return cmd_pattern(args.name, args.where, args.period, args.final, args.prereg)
        if args.command == "scan":
            return cmd_scan(args.period, args.final, args.prereg)
        if args.command == "backtest":
            return cmd_backtest(args.grid, args.hold, args.positions, not args.no_renew, args.period,
                                args.final, args.prereg, args.equity)
        if args.command == "daily":
            return cmd_daily(args.date)
        if args.command == "runs":
            return cmd_runs(args.invalidate, args.reason)
        if args.command == "events":
            return cmd_event_study()
        if args.command == "cross":
            return cross_cli.run(args, _paths, _prereg_extra)
        if args.command == "portfolio":
            return cmd_portfolio(args.config, args.period)
        if args.command == "econ":
            return cmd_econ(args.strategy, args.period)
        if args.command == "interactions":
            return cmd_interactions(args.period)
        if args.command == "audit":
            if args.audit_command == "fdr":
                return cmd_audit_fdr()
            return cmd_audit_masks() if args.audit_command == "masks" else cmd_audit()
        if args.command == "factors":
            return cmd_factors_describe()
        if args.command == "registry":
            return registry_cli.run(args, _store)
        if args.command == "ml":
            return ml_cli.run(args, _store, _paths()[0].parent)
        if args.command == "risk":
            return risk_cli.run(args, _store, _write_report, _paths()[1].parent / "cross.duckdb")
    except HoldoutLockedError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 5
    return {"status": cmd_status, "sanity": cmd_sanity, "export": cmd_export}[args.command]()


if __name__ == "__main__":
    sys.exit(main())

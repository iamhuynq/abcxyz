"""Command-line entry point: `uv run fireant <command>` (reads .env itself; real env vars win)."""

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb

from fireant_crawler.client.fireant_client import AuthError, FireAntClient, FireAntError
from fireant_crawler.client.rate_limiter import RateLimiter
from fireant_crawler.client.redact import redact
from fireant_crawler.client.token_info import inspect_token
from fireant_crawler.config import Settings, load_settings
from fireant_crawler.dotenv import read_env_file
from fireant_crawler.jobs.plans import ALL_JOBS, UPDATE_DEFAULT_JOBS, plan
from fireant_crawler.jobs.rebuild import rebuild
from fireant_crawler.jobs.runner import JobContext, NetworkDownError, run_tasks
from fireant_crawler.probe import Probe, render_report
from fireant_crawler.store.raw_store import RawStore
from fireant_crawler.store.state import CrawlState
from fireant_crawler.store.migrations import plan_schema_changes
from fireant_crawler.sessions import final_session
from fireant_crawler.store.warehouse import PRIMARY_KEYS, SchemaOutOfDateError, Warehouse
from fireant_crawler.validate.checks import render_validation_report, run_checks
from fireant_crawler.validate.contract import CHECKED_JOBS, check_files, raw_files, render_contract_section
from fireant_crawler.validate.export import export_parquet

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def _token_lines(settings: Settings, now: datetime) -> list[str]:
    info = inspect_token(settings.token or "")
    if not info.is_jwt:
        return ["- Not a JWT; expiry and scopes unknown"]
    expiry = info.expires_at.isoformat() if info.expires_at else "unknown"
    lines = [f"- Expires at (UTC): {expiry}{' (EXPIRED)' if info.is_expired(now) else ''}",
             f"- Scopes: {', '.join(info.scopes) or 'none listed'}"]
    if "orders-write" in info.scopes:
        lines.append("- WARNING: token can place orders; keep it secret. The crawler whitelist blocks order endpoints.")
    return lines


def _make_client(settings: Settings) -> FireAntClient:
    return FireAntClient(
        token=settings.token or "",
        rate_limiter=RateLimiter(settings.rate_limit_rps),
        max_retries=settings.max_retries,
        timeout_seconds=settings.timeout_seconds,
    )


def _require_token(settings: Settings) -> bool:
    if not settings.token:
        print("FIREANT_TOKEN is not set. Copy .env.example to .env and paste your token.", file=sys.stderr)
        return False
    if inspect_token(settings.token).is_expired(datetime.now(UTC)):
        print("Token is expired. Copy a fresh one into .env and retry.", file=sys.stderr)
        return False
    return True


def _warehouse_path(settings: Settings) -> Path:
    return settings.data_dir / "warehouse.duckdb"


def cmd_probe(settings: Settings, fixture: Path) -> int:
    now = datetime.now(UTC)
    token_lines = _token_lines(settings, now)
    print("\n".join(token_lines))
    with _make_client(settings) as client:
        probe = Probe(client, RawStore(settings.data_dir), today=now.date(), fetched_at=now, fixture_path=fixture)
        print(f"Probing at {settings.rate_limit_rps} req/s ...")
        probe.run()
    report_dir = settings.data_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"probe-{now.date().isoformat()}.md"
    report_path.write_text(render_report(probe, token_lines, now), encoding="utf-8")
    print(f"Done: {probe.request_count} requests. Report: {report_path}")
    return 0


EXIT_OFFLINE = 10


def preflight(client: FireAntClient) -> bool:
    """One cheap request before touching any database. False = offline / API down (not auth)."""
    try:
        client.get("/symbols/VOS")
    except AuthError:
        raise
    except FireAntError:
        return False
    return True


def cmd_crawl(settings: Settings, mode: str, jobs: list[str], symbols: list[str] | None, force: bool) -> int:
    run_at = datetime.now(VN_TZ).replace(microsecond=0)
    with _make_client(settings) as probe_client:
        if not preflight(probe_client):
            print("Offline or API unavailable (pre-flight failed). Nothing was changed; retry later.", file=sys.stderr)
            return EXIT_OFFLINE
    final = final_session(run_at)       # never fetch a session that may still be in progress
    print(f"{mode} jobs={','.join(jobs)} symbols={','.join(symbols) if symbols else 'auto'} "
          f"at {settings.rate_limit_rps} req/s, data up to the last final session {final}")
    failures = 0
    with _make_client(settings) as client, Warehouse(_warehouse_path(settings)) as wh:
        wh.ensure_schema_for_write()
        ctx = JobContext(client=client, raw=RawStore(settings.data_dir), warehouse=wh,
                         state=CrawlState(wh), run_at=run_at, today=final)
        for job in jobs:
            print(f"[{job}]")
            for batch in plan(ctx, job, mode, symbols):  # type: ignore[arg-type]
                stats = run_tasks(ctx, batch, force=force)
                failures += stats.counts["failed"]
                print(f"  [{job}] batch done: {stats.summary()}")
    print(f"Finished. Failed tasks: {failures}. See `fireant status` for details.")
    return 1 if failures else 0


def cmd_normalize(settings: Settings, dry_run: bool) -> int:
    raw = RawStore(settings.data_dir)
    if dry_run:
        counts = rebuild(raw, None)
        print("Dry run (nothing written):", dict(counts))
        return 0
    with Warehouse(_warehouse_path(settings)) as wh:
        wh.ensure_schema_for_write()
        counts = rebuild(raw, wh)
    print("Rebuilt:", dict(counts))
    return 1 if any(k.endswith(".failed") for k in counts) else 0


def cmd_status(settings: Settings) -> int:
    path = _warehouse_path(settings)
    if not path.exists():
        print(f"No warehouse at {path}")
        return 1
    with Warehouse(path, read_only=True) as wh:
        changes = wh.pending_schema_changes()
        print(f"Schema: {'up to date' if not changes else f'{len(changes)} pending change(s), run `fireant migrate --dry-run`'}")
        print("Tables:")
        for table in PRIMARY_KEYS:
            print(f"  {table}: {wh.count(table)}")
        print("Tasks (job, status, count):")
        for row in CrawlState(wh).summary():
            print(f"  {row}")
        failed = wh.connection.execute(
            "SELECT job, key, chunk, last_error FROM crawl_state WHERE status = 'failed' ORDER BY updated_at DESC LIMIT 10"
        ).fetchall()
        for row in failed:
            print(f"  FAILED {row[0]}/{row[1]}/{row[2]}: {row[3]}")
    return 0


def cmd_migrate(settings: Settings, dry_run: bool) -> int:
    path = _warehouse_path(settings)
    if not path.exists():
        print(f"No warehouse at {path}; the first backfill creates it with the current schema.")
        return 0
    with Warehouse(path, read_only=dry_run) as wh:
        changes = wh.pending_schema_changes()
        if not changes:
            print("Schema is up to date. Nothing to do.")
            return 0
        print("Pending schema changes:")
        for change in changes:
            print(f"  {change.describe()}")
        drift = [c for c in changes if not c.auto_applicable]
        if drift:
            print("Table drift cannot be applied automatically (schema.sql never alters or drops tables). "
                  "Write a manual migration with a backup and rollback plan.", file=sys.stderr)
        if dry_run:
            print("Dry run: nothing applied.")
            return 1 if drift else 0
        wh.init_schema()
        remaining = wh.pending_schema_changes()
    print(f"Applied. Remaining changes: {len(remaining)}")
    for change in remaining:
        print(f"  {change.describe()}")
    return 1 if remaining else 0


def _open_read_only(settings: Settings) -> duckdb.DuckDBPyConnection | None:
    """Read-only connection for validate/export; refuses when views/tables they rely on are out of date."""
    path = _warehouse_path(settings)
    if not path.exists():
        print(f"No warehouse at {path}", file=sys.stderr)
        return None
    con = duckdb.connect(str(path), read_only=True)
    changes = plan_schema_changes(con)
    if changes:
        con.close()
        raise SchemaOutOfDateError(changes)
    return con


def cmd_validate(settings: Settings, contract: bool = True) -> int:
    """Warehouse checks, plus the payload contract of the raw files of each job's latest fetch date
    (skipped with --no-contract, for a day when the contract itself is wrong)."""
    con = _open_read_only(settings)
    if con is None:
        return 1
    with con:
        results = run_checks(con)
        totals = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in PRIMARY_KEYS}
    now = datetime.now(VN_TZ).replace(microsecond=0)
    report = render_validation_report(results, totals, now)
    contract_errors = 0
    if contract:
        checked = check_files(raw_files(RawStore(settings.data_dir), latest_only=True))
        contract_errors = checked.errors
        report += "\n".join(render_contract_section(checked, "latest fetch date of each job")) + "\n"
    else:
        report += "\n## Contract\n\nSkipped (--no-contract).\n"
    report_dir = settings.data_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"validation-{now.date().isoformat()}.md"
    report_path.write_text(report, encoding="utf-8")
    for r in results:
        print(f"  {r.check.severity:5} {r.check.name:30} {r.count:>10,}")
    print(f"  {'error':5} {'payload_contract':30} {contract_errors:>10,}" if contract else "  contract skipped")
    print(f"Report: {report_path}")
    return 1 if contract_errors or any(r.check.severity == "error" and r.count for r in results) else 0


def cmd_contract_scan(settings: Settings, jobs: list[str], since: str | None) -> int:
    """Check stored raw files against the payload contracts (read-only)."""
    checked = check_files(raw_files(RawStore(settings.data_dir), jobs, since=since))
    print("\n".join(render_contract_section(checked, f"jobs {','.join(jobs)}" + (f", since {since}" if since else ""))))
    return 1 if checked.errors else 0


def cmd_export(settings: Settings) -> int:
    con = _open_read_only(settings)
    if con is None:
        return 1
    with con:
        written = export_parquet(con, settings.data_dir / "parquet")
    for name, rows in written.items():
        print(f"  {name}: {rows:,} rows")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fireant", description="FireAnt research crawler (GET-only)")
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe", help="Check API behaviour before crawling")
    probe.add_argument("--fixture", type=Path, default=Path("a.json"), help="Earlier VOS response to diff against")
    for name, help_text in (("backfill", "Full history (resumable)"), ("update", "Incremental daily update")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--jobs", help=f"Comma list from {','.join(ALL_JOBS)}")
        p.add_argument("--symbols", help="Comma list of symbols (default: whole universe)")
        p.add_argument("--force", action="store_true", help="Re-run tasks already marked done")
    norm = sub.add_parser("normalize", help="Rebuild warehouse from data/raw (no API calls)")
    norm.add_argument("--dry-run", action="store_true")
    sub.add_parser("status", help="Show table counts and task states (read-only)")
    mig = sub.add_parser("migrate", help="Apply schema.sql changes (new tables/views); --dry-run lists them")
    mig.add_argument("--dry-run", action="store_true")
    val = sub.add_parser("validate", help="Run data-quality checks and write a report (read-only)")
    val.add_argument("--no-contract", action="store_true", help="Skip the raw payload contract check")
    con = sub.add_parser("contract", help="Payload contract checks on stored raw files (read-only)")
    con_sub = con.add_subparsers(dest="contract_command", required=True)
    scan = con_sub.add_parser("scan", help="Check every stored raw file (or those fetched since a date)")
    scan.add_argument("--job", choices=CHECKED_JOBS, action="append", help="Repeatable; default: all")
    scan.add_argument("--since", help="Fetch date YYYY-MM-DD (inclusive)")
    sub.add_parser("export", help="Export tables and views to data/parquet (read-only)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    env_file = Path(os.environ.get("ENV_FILE", ".env"))
    settings = load_settings({**read_env_file(env_file), **os.environ})
    try:
        return _dispatch(args, settings)
    except SchemaOutOfDateError as exc:
        print(f"Refusing to run: {exc}", file=sys.stderr)
        for change in exc.changes:
            print(f"  {change.describe()}", file=sys.stderr)
        return 4


def _dispatch(args: argparse.Namespace, settings: Settings) -> int:
    if args.command == "migrate":
        return cmd_migrate(settings, args.dry_run)
    if args.command == "normalize":
        return cmd_normalize(settings, args.dry_run)
    if args.command == "status":
        return cmd_status(settings)
    if args.command == "validate":
        return cmd_validate(settings, contract=not args.no_contract)
    if args.command == "contract":
        return cmd_contract_scan(settings, args.job or list(CHECKED_JOBS), args.since)
    if args.command == "export":
        return cmd_export(settings)
    if not _require_token(settings):
        return 2
    try:
        if args.command == "probe":
            return cmd_probe(settings, args.fixture)
        default_jobs = ALL_JOBS if args.command == "backfill" else UPDATE_DEFAULT_JOBS
        jobs = [j.strip() for j in args.jobs.split(",")] if args.jobs else list(default_jobs)
        unknown = set(jobs) - set(ALL_JOBS)
        if unknown:
            print(f"Unknown jobs: {sorted(unknown)}", file=sys.stderr)
            return 2
        symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None
        return cmd_crawl(settings, args.command, jobs, symbols, args.force)
    except AuthError as exc:
        print(f"Stopped: {redact(str(exc), [settings.token])}", file=sys.stderr)
        return 3
    except NetworkDownError as exc:
        print(f"Stopped (circuit breaker): {redact(str(exc), [settings.token])}. Completed tasks are saved; "
              "re-run later to resume.", file=sys.stderr)
        return EXIT_OFFLINE
    except KeyboardInterrupt:
        print("Interrupted. Completed tasks are saved; re-run the same command to resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())

"""Tasks the UI may start: a fixed allowlist of existing CLI commands, run by a detached runner.

A run is a JSON record plus a log file in data/logs/ui/runs/. The runner (stock_ui.task_runner) is
the leader of its own process group, so a task survives closing the browser and Cancel stops the
whole group. Nothing typed on the page becomes a shell string: argv lists only.
"""

import json
import os
import re
import secrets
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

from stock_ui.db import VN_TZ
from stock_ui.locks import lock_info, pid_alive

ROOT = Path(__file__).resolve().parents[2]
STARTING_GRACE_SECONDS = 30
HISTORY_LIMIT = 50

EXIT_BUSY = 9                                       # daily.sh: another run holds the lock
EXIT_MEANINGS = {
    0: "ok",
    1: "failed (see log)",
    2: "usage or missing input",
    3: "token expired or missing",
    4: "schema out of date or build refused",
    5: "holdout locked",
    7: "latest session incomplete in the source; retry later",
    8: "validation found error-level problems; build stopped (see the validation report)",
    9: "busy: another run holds the pipeline lock; nothing done",
    10: "offline or API down; nothing changed",
}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    label: str
    help: str
    tool: str                   # "fireant", "quant" or "daily"
    args: tuple[str, ...]
    own_lock: bool = False      # daily.sh takes the pipeline lock itself
    takes_date: bool = False
    takes_run: bool = False     # requires run_id and reason (invalidate a research run)
    hidden: bool = False        # started from its own page, not from the Run tasks grid


TASKS: dict[str, TaskSpec] = {t.name: t for t in (
    TaskSpec("daily", "Full daily pipeline", "scripts/daily.sh: update, validate, build, daily report "
             "(exits at once if the latest session is already done)", "daily", (), own_lock=True),
    TaskSpec("update", "Update data", "fireant update: corporate actions and quotes up to the last final session "
             "(today only after 18:00; safe to run any time)",
             "fireant", ("update",)),
    TaskSpec("update_weekly", "Update reports and fundamentals", "fireant update --jobs report_marks,fundamental",
             "fireant", ("update", "--jobs", "report_marks,fundamental")),
    TaskSpec("validate", "Validate data", "fireant validate: data-quality report (read-only)", "fireant", ("validate",)),
    TaskSpec("status", "Crawler status", "fireant status: table counts and task states (read-only)",
             "fireant", ("status",)),
    TaskSpec("migrate_dry_run", "Schema check", "fireant migrate --dry-run: list pending schema changes",
             "fireant", ("migrate", "--dry-run")),
    TaskSpec("build", "Build features", "quant build: rebuild research.duckdb from the warehouse", "quant", ("build",)),
    TaskSpec("cross_build", "Build cross-stock data", "quant cross build: correlations, clusters and industry "
             "indexes for the Cross-stock and Market watch pages (about 40 s)", "quant", ("cross", "build")),
    TaskSpec("quant_daily", "Scan and daily report", "quant daily: scan new sessions, paper portfolio, report; "
             "with a date: re-scan that session", "quant", ("daily",), takes_date=True),
    TaskSpec("risk_paper", "Risk: evaluate paper portfolio", "quant risk evaluate --paper: limits (policy "
             "default/v2), liquidity, drawdown and stress scenarios of the current paper portfolio; writes a risk run, "
             "never changes the portfolio (results on the Risk page)", "quant", ("risk", "evaluate", "--paper")),
    TaskSpec("invalidate_run", "Invalidate research run", "quant runs --invalidate RUN --reason TEXT: mark a run "
             "invalid (kept in the log, excluded from q-values)", "quant", ("runs",), takes_run=True, hidden=True),
)}
RUN_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
REASON_MAX = 500


class TaskError(RuntimeError):
    pass


def _tool_argv(tool: str) -> list[str]:
    defaults = {"fireant": "uv run fireant", "quant": "uv run quant", "daily": f"bash {ROOT / 'scripts' / 'daily.sh'}"}
    return shlex.split(os.environ.get(f"{tool.upper()}_CMD", defaults[tool]))


def build_argv(name: str, on_date: date | str | None = None, *, run_id: str | None = None,
               reason: str | None = None) -> list[str]:
    spec = TASKS.get(name)
    if spec is None:
        raise TaskError(f"unknown task {name!r}")
    argv = _tool_argv(spec.tool) + list(spec.args)
    if on_date is not None and on_date != "":
        if not spec.takes_date:
            raise TaskError(f"task {name!r} takes no date")
        argv += ["--date", date.fromisoformat(str(on_date)).isoformat()]
    if spec.takes_run:
        reason = " ".join((reason or "").split())          # one line, no control characters
        if not run_id or not RUN_ID.match(run_id):
            raise TaskError("a valid run id is required")
        if not 3 <= len(reason) <= REASON_MAX:
            raise TaskError(f"a reason of 3 to {REASON_MAX} characters is required")
        argv += [f"--invalidate={run_id}", f"--reason={reason}"]   # '=' form: a value may start with '-' 
    elif run_id is not None or reason is not None:
        raise TaskError(f"task {name!r} takes no run id or reason")
    return argv


@dataclass
class RunRecord:
    run_id: str
    task: str
    argv: list[str]
    own_lock: bool
    started_at: str
    pid: int | None = None
    ended_at: str | None = None
    exit_code: int | None = None
    cancelled: bool = False
    refused: str | None = None

    def status(self, now: float | None = None) -> str:
        if self.refused:
            return "refused"
        if self.ended_at:
            if self.exit_code == EXIT_BUSY and not self.cancelled:
                return "refused"                    # daily.sh found the lock held: nothing ran
            if self.cancelled:
                return "cancelled"
            return "ok" if self.exit_code == 0 else "failed"
        if self.pid is None:
            started = datetime.fromisoformat(self.started_at).timestamp()
            return "starting" if (now or time.time()) - started < STARTING_GRACE_SECONDS else "interrupted"
        return "running" if pid_alive(self.pid) else "interrupted"

    def meaning(self) -> str:
        if self.refused:
            return self.refused
        if self.cancelled:
            return "cancelled"
        if self.exit_code is None:
            return ""
        return EXIT_MEANINGS.get(self.exit_code, f"exit {self.exit_code}")

    def duration(self) -> float | None:
        if not self.ended_at:
            return None
        return (datetime.fromisoformat(self.ended_at) - datetime.fromisoformat(self.started_at)).total_seconds()


def runs_dir(data_dir: Path) -> Path:
    return data_dir / "logs" / "ui" / "runs"


def record_path(data_dir: Path, run_id: str) -> Path:
    return runs_dir(data_dir) / f"{run_id}.json"


def log_path(data_dir: Path, run_id: str) -> Path:
    return runs_dir(data_dir) / f"{run_id}.log"


def load_record(data_dir: Path, run_id: str) -> RunRecord:
    return RunRecord(**json.loads(record_path(data_dir, run_id).read_text()))


def create_record(data_dir: Path, rec: RunRecord) -> None:
    """Exclusive create: two starts can never share a record."""
    with record_path(data_dir, rec.run_id).open("x") as fh:
        fh.write(json.dumps(asdict(rec), indent=1))


def save_record(data_dir: Path, rec: RunRecord) -> None:
    path = record_path(data_dir, rec.run_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(rec), indent=1))
    os.replace(tmp, path)


def history(data_dir: Path, limit: int = HISTORY_LIMIT) -> list[RunRecord]:
    files = sorted(runs_dir(data_dir).glob("*.json"), reverse=True)[:limit]
    out = []
    for f in files:
        try:
            out.append(RunRecord(**json.loads(f.read_text())))
        except (ValueError, TypeError):
            continue
    return out


def active_run(data_dir: Path) -> RunRecord | None:
    for rec in history(data_dir, limit=10):
        if rec.status() in ("starting", "running"):
            return rec
    return None


def busy_reason(data_dir: Path) -> str | None:
    """Why no task may start now, or None."""
    rec = active_run(data_dir)
    if rec is not None:
        return f"UI task '{rec.task}' is {rec.status()} (run {rec.run_id})"
    info = lock_info(data_dir)
    if info is not None and not info.stale:
        return f"{info.owner} holds the pipeline lock (pid {info.pid}, {info.age_seconds / 60:.0f} min)"
    return None


def start(data_dir: Path, name: str, on_date: date | str | None = None, *, run_id: str | None = None,
          reason: str | None = None) -> str:
    argv = build_argv(name, on_date, run_id=run_id, reason=reason)
    reason = busy_reason(data_dir)
    if reason:
        raise TaskError(reason)
    now = datetime.now(VN_TZ)
    ui_run = f"{now:%Y%m%d-%H%M%S}-{name}-{secrets.token_hex(2)}"
    runs_dir(data_dir).mkdir(parents=True, exist_ok=True)
    create_record(data_dir, RunRecord(ui_run, name, argv, TASKS[name].own_lock, now.isoformat(timespec="seconds")))
    subprocess.Popen([sys.executable, "-m", "stock_ui.task_runner", str(data_dir.resolve()), ui_run],
                     cwd=ROOT, start_new_session=True, stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return ui_run


def cancel(data_dir: Path, run_id: str) -> bool:
    rec = load_record(data_dir, run_id)
    if rec.status() != "running" or rec.pid is None:
        return False
    try:
        os.killpg(rec.pid, signal.SIGTERM)  # the runner leads its own process group
    except (ProcessLookupError, PermissionError):
        return False                        # it ended in the meantime
    return True


def tail(data_dir: Path, run_id: str, max_bytes: int = 20_000) -> str:
    path = log_path(data_dir, run_id)
    if not path.exists():
        return ""
    with path.open("rb") as fh:
        size = fh.seek(0, os.SEEK_END)
        fh.seek(max(0, size - max_bytes))
        return fh.read().decode("utf-8", errors="replace")

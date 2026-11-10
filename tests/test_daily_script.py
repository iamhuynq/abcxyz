"""scripts/daily.sh with stub commands: ordering, exit codes, marker, lock. No API, no real data."""

import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "daily.sh"
STUB = """import os, sys
name, args = sys.argv[1], " ".join(sys.argv[2:])
with open(os.environ["STUB_LOG"], "a") as fh:
    fh.write(f"{name} {args}\\n")
# exit code from RC_<NAME>_<ARG1>[_<ARG2>], e.g. RC_FIREANT_UPDATE or RC_FIREANT_UPDATE___JOBS (weekly)
sys.exit(int(os.environ.get(f"RC_{name}_{'_'.join(sys.argv[2:4])}".upper().replace("-", "_"), "0")))
"""


def run(tmp_path, target="2026-10-05", weekly_done="2026-10-02", **rc):
    """weekly_done: content of data/.weekly_done before the run (None: no marker)."""
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    if weekly_done and not (data / ".weekly_done").exists():
        (data / ".weekly_done").write_text(weekly_done)
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    env = {**os.environ, "DATA_DIR": str(tmp_path / "data"), "TARGET_DAY": target,
           "FIREANT_CMD": f"{sys.executable} {stub} fireant", "QUANT_CMD": f"{sys.executable} {stub} quant",
           "NOTIFY_CMD": "true", "STUB_LOG": str(tmp_path / "calls.log")}
    env |= {f"RC_{k.upper()}": str(v) for k, v in rc.items()}
    proc = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60)
    calls = (tmp_path / "calls.log").read_text().splitlines() if (tmp_path / "calls.log").exists() else []
    return proc.returncode, calls, proc.stdout


def test_success_runs_steps_in_order_writes_marker_and_next_trigger_exits(tmp_path):
    rc, calls, _ = run(tmp_path)
    assert rc == 0
    assert calls == ["fireant update", "fireant validate", "quant build", "quant daily"]   # Monday target
    assert (tmp_path / "data" / ".daily_done").read_text().strip() == "2026-10-05"
    (tmp_path / "calls.log").unlink()
    rc, calls, out = run(tmp_path)
    assert rc == 0 and calls == [] and "Already done" in out
    assert not (tmp_path / "data" / ".daily.lock").exists()


def test_offline_stops_before_build_and_writes_no_marker(tmp_path):
    rc, calls, _ = run(tmp_path, fireant_update=10)
    assert rc == 10 and calls == ["fireant update"]
    assert not (tmp_path / "data" / ".daily_done").exists()


def test_expired_token_exit_code(tmp_path):
    rc, _, _ = run(tmp_path, fireant_update=3)
    assert rc == 3


def test_incomplete_latest_session_is_retried_later(tmp_path):
    rc, calls, _ = run(tmp_path, quant_daily=7)
    assert rc == 7 and calls[-1].startswith("quant daily")
    assert not (tmp_path / "data" / ".daily_done").exists()


def test_validation_errors_stop_before_the_build(tmp_path):
    """fireant validate exits 1 only for error-level findings (warn/info exit 0): no build, no marker."""
    rc, calls, _ = run(tmp_path, fireant_validate=1)
    assert rc == 8 and calls[-1].startswith("fireant validate")
    assert not (tmp_path / "data" / ".daily_done").exists()
    rc, calls, _ = run(tmp_path)                                   # fixed: the next trigger runs everything
    assert rc == 0 and calls[-1].startswith("quant daily")


def test_live_lock_makes_a_second_run_exit_and_stale_lock_is_replaced(tmp_path):
    lock = tmp_path / "data" / ".daily.lock"
    lock.mkdir(parents=True)
    (lock / "pid").write_text(str(os.getpid()))                    # a live process holds the lock
    rc, calls, out = run(tmp_path)
    assert rc == 9 and calls == [] and "in progress" in out         # refused, not "done"
    (lock / "pid").write_text("999999")                             # dead pid: stale
    rc, calls, _ = run(tmp_path)
    assert rc == 0 and calls and not lock.exists()


def test_friday_target_adds_weekly_jobs_even_when_caught_up_later(tmp_path):
    rc, calls, _ = run(tmp_path, target="2026-10-09")               # a Friday session
    assert rc == 0 and calls[1] == "fireant update --jobs report_marks,fundamental"


WEEKLY = "fireant update --jobs report_marks,fundamental"


def test_weekly_jobs_catch_up_when_the_last_weekly_update_is_a_week_old(tmp_path):
    rc, calls, _ = run(tmp_path, target="2026-10-12", weekly_done="2026-10-02")    # Monday, Friday 10-09 missed
    assert rc == 0 and calls[1] == WEEKLY
    assert (tmp_path / "data" / ".weekly_done").read_text().strip() == "2026-10-12"


def test_weekly_jobs_skipped_when_recent_and_target_is_not_friday(tmp_path):
    rc, calls, _ = run(tmp_path, target="2026-10-07", weekly_done="2026-10-02")    # Wednesday, 5 days later
    assert rc == 0 and WEEKLY not in calls


def test_weekly_jobs_run_without_a_marker_and_retry_after_a_failure(tmp_path):
    rc, calls, _ = run(tmp_path, target="2026-10-07", weekly_done=None)
    assert rc == 0 and calls[1] == WEEKLY
    (tmp_path / "data" / ".weekly_done").unlink()
    (tmp_path / "data" / ".daily_done").unlink()
    (tmp_path / "calls.log").unlink()
    rc, calls, _ = run(tmp_path, target="2026-10-07", weekly_done=None, fireant_update___jobs=1)
    assert rc == 0 and calls[1] == WEEKLY and not (tmp_path / "data" / ".weekly_done").exists()


def test_script_uses_no_bsd_only_date_or_stat_flags():
    """CI runs on Linux (GNU coreutils): `date -j`, `date -v` and `stat -f` exist only on macOS / BSD."""
    import re
    text = SCRIPT.read_text()
    assert not re.search(r"\bdate\s+-[jv]|\bstat\s+-f\b", text), "BSD-only date/stat flags in daily.sh"

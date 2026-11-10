"""UI task runner with stub commands: allowlist, records, exit codes, shared lock, cancel. No real CLI."""

import os
import sys
import time

import pytest

from stock_ui import locks, tasks

STUB = """import os, sys, time
print("stub", " ".join(sys.argv[1:]), flush=True)
if "sleep" in sys.argv:
    time.sleep(30)
sys.exit(int(os.environ.get("STUB_RC", "0")))
"""


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    monkeypatch.setenv("FIREANT_CMD", f"{sys.executable} {stub} fireant")
    monkeypatch.setenv("QUANT_CMD", f"{sys.executable} {stub} quant")
    monkeypatch.setenv("DAILY_CMD", f"{sys.executable} {stub} daily")
    return tmp_path / "data"


def wait_for(data_dir, run_id, states, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            rec = tasks.load_record(data_dir, run_id)
            if rec.status() in states:
                return rec
        except FileNotFoundError:
            pass
        time.sleep(0.1)
    raise AssertionError(f"{run_id} never reached {states}: {tasks.load_record(data_dir, run_id)}")


def test_allowlist_and_date_validation():
    with pytest.raises(tasks.TaskError):
        tasks.build_argv("update; touch pwned")
    with pytest.raises(tasks.TaskError):
        tasks.build_argv("update", "2026-10-05")                   # takes no date
    with pytest.raises(ValueError):
        tasks.build_argv("quant_daily", "2026-10-05; touch pwned")
    assert tasks.build_argv("quant_daily", "2026-10-05")[-2:] == ["--date", "2026-10-05"]


def test_run_records_exit_code_log_and_releases_lock(data_dir):
    run_id = tasks.start(data_dir, "update")
    rec = wait_for(data_dir, run_id, {"ok", "failed"})
    assert rec.status() == "ok" and rec.exit_code == 0 and rec.meaning() == "ok"
    assert "stub fireant update" in tasks.tail(data_dir, run_id)
    assert not locks.lock_path(data_dir).exists()
    assert [r.run_id for r in tasks.history(data_dir)] == [run_id]


def test_exit_code_meaning(data_dir, monkeypatch):
    monkeypatch.setenv("STUB_RC", "10")
    rec = wait_for(data_dir, tasks.start(data_dir, "update"), {"ok", "failed"})
    assert rec.exit_code == 10 and "offline" in rec.meaning()


def test_busy_exit_code_is_reported_as_refused_not_failed(data_dir, monkeypatch):
    """daily.sh exits 9 when another run holds the lock: nothing ran, so the UI shows refused."""
    monkeypatch.setenv("STUB_RC", "9")
    rec = wait_for(data_dir, tasks.start(data_dir, "update"), {"ok", "failed", "refused"})
    assert rec.exit_code == 9 and rec.status() == "refused" and "busy" in rec.meaning()


def test_refuses_while_a_live_pipeline_holds_the_lock(data_dir):
    lock = locks.lock_path(data_dir)
    lock.mkdir(parents=True)
    (lock / "pid").write_text(str(os.getpid()))                    # a live process: like daily.sh running
    with pytest.raises(tasks.TaskError, match="daily.sh holds the pipeline lock"):
        tasks.start(data_dir, "build")
    assert lock.exists()


def test_takes_over_a_stale_lock(data_dir):
    lock = locks.lock_path(data_dir)
    lock.mkdir(parents=True)
    (lock / "pid").write_text("999999")                            # dead process
    rec = wait_for(data_dir, tasks.start(data_dir, "build"), {"ok", "failed"})
    assert rec.status() == "ok" and not lock.exists()


def test_daily_sh_task_does_not_take_the_lock_itself(data_dir):
    rec = wait_for(data_dir, tasks.start(data_dir, "daily"), {"ok", "failed"})
    assert rec.status() == "ok" and "stub daily" in tasks.tail(data_dir, rec.run_id)


def test_only_one_task_at_a_time_and_cancel(data_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("QUANT_CMD", f"{sys.executable} {tmp_path / 'stub.py'} quant sleep")
    run_id = tasks.start(data_dir, "build")
    wait_for(data_dir, run_id, {"running"})
    assert locks.lock_info(data_dir).owner == f"ui:{run_id}"
    with pytest.raises(tasks.TaskError, match="is running"):
        tasks.start(data_dir, "update")
    assert tasks.cancel(data_dir, run_id)
    rec = wait_for(data_dir, run_id, {"cancelled", "ok", "failed"}, timeout=10)
    assert rec.status() == "cancelled"
    assert not locks.lock_path(data_dir).exists()
    assert tasks.busy_reason(data_dir) is None


def test_runner_that_died_is_reaped_and_does_not_block(data_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("QUANT_CMD", f"{sys.executable} {tmp_path / 'stub.py'} quant sleep")
    run_id = tasks.start(data_dir, "build")
    rec = wait_for(data_dir, run_id, {"running"})
    os.kill(rec.pid, 9)                                            # runner killed hard: no end record, lock left
    os.killpg(rec.pid, 9)                                          # (and its orphaned stub, for test hygiene)
    wait_for(data_dir, run_id, {"interrupted"}, timeout=10)        # reaped, not a zombie "running" forever
    assert locks.lock_info(data_dir).stale
    assert tasks.busy_reason(data_dir) is None
    rec2 = wait_for(data_dir, tasks.start(data_dir, "update"), {"ok", "failed"})
    assert rec2.status() == "ok"                                   # stale lock taken over


def test_young_lock_without_pid_is_held(data_dir):
    lock = locks.lock_path(data_dir)
    lock.mkdir(parents=True)                                       # daily.sh between mkdir and writing pid
    assert not locks.acquire(data_dir, "ui:test")
    assert lock.exists()


def test_cancel_of_a_finished_run_returns_false(data_dir):
    rec = wait_for(data_dir, tasks.start(data_dir, "update"), {"ok", "failed"})
    assert tasks.cancel(data_dir, rec.run_id) is False


def test_invalidate_requires_valid_run_id_and_reason():
    with pytest.raises(tasks.TaskError):
        tasks.build_argv("invalidate_run", run_id="run-a", reason="")
    with pytest.raises(tasks.TaskError):
        tasks.build_argv("invalidate_run", run_id="a b; touch pwned", reason="because")
    with pytest.raises(tasks.TaskError):
        tasks.build_argv("update", run_id="run-a", reason="because")
    argv = tasks.build_argv("invalidate_run", run_id="run-a", reason="  two\nlines  ")
    assert argv[-2:] == ["--invalidate=run-a", "--reason=two lines"]


def test_invalidate_values_starting_with_a_dash_reach_the_cli_intact():
    from quant_research import cli
    seen = {}
    argv = tasks.build_argv("invalidate_run", run_id="-run", reason="--bug in definition")
    real = cli.cmd_runs
    cli.cmd_runs = lambda run, reason: seen.update(run=run, reason=reason) or 0
    try:
        assert cli.main(argv[argv.index("runs"):]) == 0
    finally:
        cli.cmd_runs = real
    assert seen == {"run": "-run", "reason": "--bug in definition"}
    assert "invalidate_run" in tasks.TASKS and tasks.TASKS["invalidate_run"].hidden

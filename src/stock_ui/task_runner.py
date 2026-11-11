"""Detached runner for one UI task: `python -m stock_ui.task_runner <data_dir> <run_id>`.

Started by stock_ui.tasks.start as the leader of a new process group. Takes the pipeline lock
(unless the task is daily.sh, which takes it itself), runs the command with output to the run log,
records the exit code, and releases the lock. SIGTERM (Cancel) reaches the whole group; the runner
itself keeps going until the command has exited, so the record and the lock are always cleaned up.
"""

import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from stock_ui import locks
from stock_ui.db import VN_TZ
from stock_ui.tasks import ROOT, load_record, log_path, save_record


def main(argv: list[str]) -> int:
    data_dir, run_id = Path(argv[0]), argv[1]
    rec = load_record(data_dir, run_id)
    cancelled = False

    def on_term(_signum, _frame) -> None:
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGTERM, on_term)
    owns_lock = False
    if not rec.own_lock:
        owns_lock = locks.acquire(data_dir, f"ui:{run_id}")
        if not owns_lock:
            info = locks.lock_info(data_dir)
            rec.refused = f"busy: {info.owner if info else 'another run'} holds the pipeline lock"
            rec.ended_at = _now()
            save_record(data_dir, rec)
            return 1
    try:
        rec.pid = os.getpid()
        save_record(data_dir, rec)
        with log_path(data_dir, run_id).open("ab") as log:
            log.write(f"$ {' '.join(rec.argv)}\n".encode())
            log.flush()
            try:
                # DATA_DIR pinned: the command (and daily.sh's lock) must use the directory the UI locked
                proc = subprocess.Popen(rec.argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                        stderr=subprocess.STDOUT, env={**os.environ, "DATA_DIR": str(data_dir)})
                rc = proc.wait()
            except OSError as exc:
                log.write(f"could not start: {exc}\n".encode())
                rc = 127
        rec.exit_code, rec.cancelled, rec.ended_at = rc, cancelled, _now()
        # Release before the record says "ended": a reader that sees the end must also see the lock gone.
        # Until the save, the record still shows this run as running, so no other task can start.
        if owns_lock:
            locks.release(data_dir)
            owns_lock = False
        save_record(data_dir, rec)
        return 0
    finally:
        if owns_lock:
            locks.release(data_dir)


def _now() -> str:
    return datetime.now(VN_TZ).isoformat(timespec="seconds")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

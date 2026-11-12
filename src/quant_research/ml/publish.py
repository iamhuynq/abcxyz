"""Atomic, never-overwriting directory artifacts behind a per-path OS lock (docs/ml-plan.md, PR #11 and PR #12).

Shared by the dataset artifact (`dataset.write_dataset`) and the model artifact (`artifacts.save_model`). An
artifact is written into a temporary sibling directory, verified by reading it back, then renamed into place in
one step; on any failure the temporary directory is removed and nothing is published.
"""

import fcntl
import json
import os
import shutil
import socket
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path


def acquire(lock: Path, path: Path) -> int:
    """One publisher per path, with an OS advisory lock (fcntl.flock, exclusive, non-blocking) on a lock file that
    is never deleted. The kernel releases the lock when its holder exits, crashes or is killed, so there is no stale
    lock to recover and no PID bookkeeping. The file is never unlinked: deleting a flock file while another process
    holds it open would let two publishers lock different inodes. Returns the open descriptor; closing it unlocks.
    POSIX only (macOS, Linux)."""
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise FileExistsError(f"{path} is being published by another process (lock {lock})") from None
    try:                                              # diagnostics only: who holds it; nothing reads it
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "since": time.time()}).encode())
    except BaseException:
        os.close(fd)
        raise
    return fd


@contextmanager
def locked(path: Path) -> Iterator[Path]:
    """Hold the publish lock of `path` (`.{name}.lock` next to it) for the duration of the block."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = acquire(path.with_name(f".{path.name}.lock"), path)
    try:
        yield path
    finally:
        os.close(fd)                                  # releases the flock; the lock file stays


def publish_dir(path: Path, write: Callable[[Path], None], verify: Callable[[Path], object], kind: str) -> Path:
    """Publish a directory at `path`, which must not exist: `write(tmp)` fills a temporary sibling, `verify(tmp)`
    reads it back (and raises if it is wrong), then the directory is renamed into place. Call under `locked`."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists: {kind} is never overwritten")
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    tmp.mkdir()
    try:
        write(tmp)
        verify(tmp)                                   # verify every file before it becomes visible
        tmp.rename(path)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return path

"""Immutable raw response store: data/raw/{job}/{key}/{fetch_date}/{chunk}__{HHMMSS}[_NNN].json.gz

The run time in the file name keeps same-day re-runs apart. A file is never replaced: it is created with
os.link, which fails when the name exists, and a second write in the same second gets the next suffix
(_001, _002, ...), so the names still sort in write order.
"""

import gzip
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fireant_crawler.client.fireant_client import ApiResponse

MAX_SAME_SECOND = 1000
_SAFE_NAME = re.compile(r"[A-Za-z0-9_.=-]{1,120}")


def _check_name(kind: str, value: str) -> str:
    if not _SAFE_NAME.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"Unsafe {kind} for raw store path: {value!r}")
    return value


class RawStore:
    def __init__(self, data_dir: Path) -> None:
        self._root = data_dir / "raw"

    def write(self, job: str, key: str, chunk: str, response: ApiResponse, fetched_at: datetime) -> Path:
        directory = (
            self._root
            / _check_name("job", job)
            / _check_name("key", key)
            / fetched_at.date().isoformat()
        )
        directory.mkdir(parents=True, exist_ok=True)
        base = f"{_check_name('chunk', chunk)}__{fetched_at:%H%M%S}"

        envelope = {
            "fetched_at": fetched_at.isoformat(),
            "request": {"path": response.path, "params": response.params},
            "status": response.status,
            "data": response.data,
        }
        tmp = directory / f".{uuid.uuid4().hex}.tmp"
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(envelope, fh, ensure_ascii=False)
        try:
            for n in range(MAX_SAME_SECOND):
                target = directory / (f"{base}.json.gz" if n == 0 else f"{base}_{n:03d}.json.gz")
                try:
                    os.link(tmp, target)                   # exclusive: never replaces an existing file
                    return target
                except FileExistsError:
                    continue
            raise FileExistsError(f"{MAX_SAME_SECOND} raw files named {base} already exist")
        finally:
            tmp.unlink(missing_ok=True)

    def iter_files(self, job: str) -> list[tuple[str, Path]]:
        """(key, path) for every raw file of a job, oldest fetch first."""
        base = self._root / _check_name("job", job)
        if not base.is_dir():
            return []
        files = [(p.parts[-3], p) for p in base.glob("*/*/*.json.gz")]
        return sorted(files, key=lambda kp: (kp[1].parts[-2], kp[1].name.rsplit("__", 1)[-1], kp[1].name))

    @staticmethod
    def read(path: Path) -> dict[str, Any]:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)

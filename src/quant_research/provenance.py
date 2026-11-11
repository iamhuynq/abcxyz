"""Reproducibility metadata for every run and report (docs/provenance-exposure-plan.md, part 1).

A run is traceable to: git commit (+ whether src/ had uncommitted changes), dataset (warehouse file hash
recorded by `quant build`), feature build, code hash, the hash of the run configuration and, from 2026-11-11, the
Python and numerical library versions (`env`) and the selection rule that pattern versions do not hash
(`selection`). Older runs have neither; readers use `.get`.
"""

import hashlib
import json
import platform
import subprocess
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

from quant_research.patterns import DECILE_TIE_BREAK
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def git_info() -> tuple[str, bool | None]:
    """(commit SHA or 'unknown', True if src/ has uncommitted changes, None if unknown)."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                             timeout=10, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=REPO, capture_output=True,
                                text=True, timeout=10, check=True).stdout
        return sha or "unknown", bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        return "unknown", None


# Implementation rules that a pattern version (a hash of the condition) does not capture. Change with the code.
SELECTION = f"deciles v2: ties by {DECILE_TIE_BREAK}"   # v1 (before 2026-11-11): ties in physical row order (H-2)
ENV_PACKAGES = ("duckdb", "numpy", "pandas", "scipy", "statsmodels")


@lru_cache(maxsize=1)
def environment() -> str:
    """Python and the libraries that compute results, e.g. 'python 3.12.13, duckdb 1.5.6, numpy 2.5.3, ...'."""
    parts = [f"python {platform.python_version()}"]
    for name in ENV_PACKAGES:
        try:
            parts.append(f"{name} {version(name)}")
        except PackageNotFoundError:
            parts.append(f"{name} unknown")
    return ", ".join(parts)


def config_hash(config_json: str) -> str:
    return hashlib.sha256(config_json.encode()).hexdigest()[:16]


def collect(con, research_code_hash: str, config_json: str | None = None) -> dict:
    """Provenance dict; `con` has research.duckdb attached as `rs` (warehouse hash from the latest build)."""
    sha, dirty = git_info()
    build_id = warehouse_sha = None
    try:
        row = con.execute("""SELECT build_id, warehouse_sha FROM rs.feature_builds
                             ORDER BY built_at DESC LIMIT 1""").fetchone()
        if row:
            build_id, warehouse_sha = row
    except Exception:              # older research.duckdb without the column: report what is known
        row = con.execute("SELECT build_id FROM rs.feature_builds ORDER BY built_at DESC LIMIT 1").fetchone()
        build_id = row[0] if row else None
    out = {"git_sha": sha, "git_dirty": dirty, "warehouse_sha": warehouse_sha, "feature_build_id": build_id,
           "code_hash": research_code_hash,      # key kept for older runs: it is the research_code_hash
           "env": environment(),
           "selection": SELECTION}
    if config_json is not None:
        out["config_hash"] = config_hash(config_json)
    return out


def line(p: dict) -> str:
    """One markdown line for report headers."""
    dirty = {True: " (uncommitted changes in src/)", False: "", None: " (git status unknown)"}[p.get("git_dirty")]
    sha = (p.get("git_sha") or "unknown")[:10]
    wh = (p.get("warehouse_sha") or "unknown")[:12]
    return (f"- Provenance: commit `{sha}`{dirty}, warehouse `{wh}`, feature build `{p.get('feature_build_id')}`, "
            f"code `{p.get('code_hash')}`" + (f", config `{p['config_hash']}`" if p.get("config_hash") else "")
            + (f"; {p['env']}" if p.get("env") else ""))


def from_runs(con, run_ids: list[str]) -> list[str]:
    """Report lines with the stored provenance of the given runs (one line per distinct provenance)."""
    if not run_ids:
        return []
    marks = ", ".join("?" for _ in run_ids)
    seen, out = set(), []
    for (params,) in con.execute(f"SELECT params FROM research_runs WHERE run_id IN ({marks}) ORDER BY run_id",
                                 run_ids).fetchall():
        p = json.loads(params).get("provenance")
        if p:
            p = {k: v for k, v in p.items() if k != "config_hash"}
            key = json.dumps(p, sort_keys=True)
            if key not in seen:
                seen.add(key)
                out.append(line(p))
    return out

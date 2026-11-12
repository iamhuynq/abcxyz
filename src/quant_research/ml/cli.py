"""`quant ml run EXPERIMENT.json`: one walk-forward experiment, one logged primary test (docs/ml-plan.md, PR #13)."""

import argparse
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from quant_research.ml import experiment_file, runner
from quant_research.ml.contracts import ContractError
from quant_research.results import ResultsStore


def add_parser(sub: argparse._SubParsersAction) -> None:
    ml = sub.add_parser("ml", help="ML research layer (research period only)")
    ms = ml.add_subparsers(dest="ml_command", required=True)
    rn = ms.add_parser("run", help="Run one walk-forward experiment (logs exactly one primary test)")
    rn.add_argument("experiment", type=Path, help="Experiment JSON: every field explicit")
    rn.add_argument("--runs-dir", type=Path, help="Run artifacts (default: <data dir>/ml/runs)")


def run(args: argparse.Namespace, store_factory: Callable[[], ResultsStore], data_dir: Path) -> int:
    try:
        spec = experiment_file.load(args.experiment)
    except ContractError as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    runs_dir = args.runs_dir or data_dir / "ml" / "runs"
    now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).replace(microsecond=0)
    with store_factory() as store:
        try:
            run_id, primary = runner.run_experiment(store, spec, runs_dir, now)
        except runner.ExperimentRefused as exc:
            print(f"Refused (nothing logged): {exc}", file=sys.stderr)
            return 2
    print(f"run_id: {run_id}\nstatus: ok\nexperiment_id: {spec.experiment_id}\n"
          f"primary test: {primary['test']} on {primary['n_dates']} dates, mean {primary['mean']:.5f}, "
          f"p {primary['p']:.4g} (lag {primary['lag']})\nartifacts: {runs_dir / run_id}")
    return 0

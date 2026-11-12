"""ML research contracts (docs/ml-plan.md sections 3, 4 and 6). Frozen specs with validators and versioned hashes.

Every spec validates itself when it is built: an invalid spec cannot exist. The research cutoff (the end of the
research period, read from backtest.runner.PERIOD_RANGES) is an invariant: no dataset, split or fold may reach
past it. Validation and holdout are not available to ML; a model is confirmed only by a pre-registration on
forward data.
"""

import re
from dataclasses import dataclass, field
from datetime import date, datetime

from quant_research import factors
from quant_research.backtest.runner import PERIOD_RANGES
from quant_research.ml import allowlist
from quant_research.ml.canonical import digest
from quant_research.results import ResearchParams

RESEARCH_START, RESEARCH_END = PERIOD_RANGES["research"]
LABEL_COLUMNS = {"fwd_ret_exec_3d": 3, "fwd_ret_exec_5d": 5, "fwd_ret_exec_10d": 10, "fwd_ret_exec_20d": 20,
                 "fwd_excess_exec_5d": 5, "fwd_excess_exec_10d": 10, "fwd_excess_exec_20d": 20}
UNIVERSES = {"point_in_time_default": ResearchParams().point_in_time_sql()}   # known at the close of t
MODELS = ("ridge", "elasticnet")                 # PR #12 adds a model only through this list
PRIMARY_TESTS = ("incremental_rank_ic_newey_west",)


class ContractError(ValueError):
    pass


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ContractError(message)


def _check_date(name: str, d) -> None:
    _require(isinstance(d, date) and not isinstance(d, datetime), f"{name} must be a date, not {type(d).__name__}")
    _require(d <= RESEARCH_END, f"{name} {d} is after the research cutoff {RESEARCH_END}: validation and holdout "
                                "are not available to ML")


@dataclass(frozen=True)
class FeatureSpec:
    table: str
    column: str
    source_version: str
    availability: str = allowlist.AVAILABILITY

    def __post_init__(self) -> None:
        why = allowlist.refusal(self.table, self.column, self.source_version, self.availability)
        _require(not why, f"feature {self.table}.{self.column}: {why}")

    @property
    def name(self) -> str:
        return f"{self.table}.{self.column}"

    @property
    def version(self) -> str:
        return digest(self)

    @classmethod
    def current(cls, table: str, column: str) -> "FeatureSpec":
        """An allowlisted feature at the current source version."""
        return cls(table, column, allowlist.source_version(table))


@dataclass(frozen=True)
class LabelSpec:
    column: str
    horizon: int
    entry: str = "open_t+1"          # the existing executable convention (stock_targets)
    exit: str = "close_t+h"
    costs: str = "gross"             # costs apply at the portfolio stage (cost model v1)
    mask_window: str = "horizon"     # the per-horizon outcome masks: price jump, entry blocked, period crossing

    def __post_init__(self) -> None:
        _require(self.column in LABEL_COLUMNS, f"label {self.column!r} is not an executable stock_targets label")
        _require(LABEL_COLUMNS[self.column] == self.horizon,
                 f"label {self.column} has horizon {LABEL_COLUMNS[self.column]}, not {self.horizon}")
        _require((self.entry, self.exit, self.costs, self.mask_window) == ("open_t+1", "close_t+h", "gross", "horizon"),
                 "label convention must be entry open_t+1, exit close_t+h, gross of costs, per-horizon masks")

    @property
    def version(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class DatasetSpec:
    features: tuple[FeatureSpec, ...]
    label: LabelSpec
    period_start: date
    period_end: date
    universe: str = "point_in_time_default"

    def __post_init__(self) -> None:
        _require(len(self.features) > 0, "a dataset needs at least one feature")
        names = [f.name for f in self.features]
        _require(len(set(names)) == len(names), f"duplicate features: {sorted({n for n in names if names.count(n) > 1})}")
        _require(self.universe in UNIVERSES, f"unknown universe {self.universe!r}")
        _check_date("period_start", self.period_start)
        _check_date("period_end", self.period_end)
        _require(RESEARCH_START <= self.period_start <= self.period_end, "period must be ordered and in research")

    @property
    def fingerprint(self) -> str:
        """Hash of the definition, the universe SQL included. The data identity (warehouse and feature build) is
        added by the dataset builder (PR #11)."""
        return digest({"spec": self, "universe_sql": UNIVERSES[self.universe]})


@dataclass(frozen=True)
class SplitSpec:
    kind: str                        # "expanding" or "rolling"; there is no shuffle, by construction
    train_min_sessions: int
    test_sessions: int
    step_sessions: int
    embargo_sessions: int
    last_date: date
    train_sessions: int | None = None    # rolling window length
    min_train_rows: int = 1

    def __post_init__(self) -> None:
        _require(self.kind in ("expanding", "rolling"), f"split kind {self.kind!r} is not chronological")
        _require(self.kind == "expanding" or (self.train_sessions or 0) >= self.train_min_sessions,
                 "a rolling split needs train_sessions >= train_min_sessions")
        _require(self.kind == "rolling" or self.train_sessions is None, "train_sessions is for rolling splits only")
        for name in ("train_min_sessions", "test_sessions", "step_sessions", "min_train_rows"):
            _require(isinstance(getattr(self, name), int) and getattr(self, name) > 0, f"{name} must be > 0")
        _require(isinstance(self.embargo_sessions, int) and self.embargo_sessions >= 0, "embargo_sessions must be >= 0")
        _check_date("last_date", self.last_date)

    @property
    def version(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class ExperimentSpec:
    hypothesis: str
    dataset: DatasetSpec
    split: SplitSpec
    model: str
    seed: int
    baseline: str                    # a Factor Engine factor: its rank is the benchmark score
    primary_test: str
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(bool(self.hypothesis.strip()), "an experiment needs a written hypothesis")
        _require(self.model in MODELS, f"model {self.model!r} is not an approved family {MODELS}")
        _require(isinstance(self.seed, int) and self.seed >= 0, "an experiment needs a seed >= 0")
        _require(self.baseline in factors.FACTORS, f"baseline {self.baseline!r} is not a Factor Engine factor")
        _require(self.primary_test in PRIMARY_TESTS, f"primary test {self.primary_test!r} is not declared")
        _require(self.split.last_date >= self.dataset.period_start, "the split ends before the dataset starts")
        _require(all(re.fullmatch(r"[a-z_]+", k) for k in self.params), "parameter names must be lower snake case")
        digest(self.params)          # refuses parameters without a canonical form

    @property
    def experiment_id(self) -> str:
        return digest(self)

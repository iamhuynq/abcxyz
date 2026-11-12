"""ML research contracts (docs/ml-plan.md sections 3, 4 and 6). Frozen specs with validators and versioned hashes.

Every spec validates itself when it is built: an invalid spec cannot exist. The research cutoff (the end of the
research period, read from backtest.runner.PERIOD_RANGES) is an invariant: no dataset, split or fold may reach
past it. Validation and holdout are not available to ML; a model is confirmed only by a pre-registration on
forward data.
"""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from types import MappingProxyType

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


def _check_type(name: str, value, types: type | tuple, positive: bool = False) -> None:
    ok = isinstance(value, types) and not (isinstance(value, bool) and bool not in (types if isinstance(types, tuple)
                                                                                     else (types,)))
    _require(ok, f"{name} must be {getattr(types, '__name__', types)}, not {type(value).__name__}")
    if positive:
        _require(value > 0, f"{name} must be > 0")


def freeze(x):
    """An immutable copy: mappings become read-only mappings of frozen values, lists and tuples become tuples;
    only None, bool, int, finite float, str and date are accepted as leaves. The caller's objects are never kept,
    so mutating them later cannot change a spec or its hash."""
    if x is None or isinstance(x, (bool, int, str, date)):
        return x
    if isinstance(x, float):
        _require(math.isfinite(x), f"non-finite parameter value {x!r}")
        return x
    if isinstance(x, Mapping):
        _require(all(isinstance(k, str) for k in x), "parameter names must be strings")
        return MappingProxyType({k: freeze(v) for k, v in x.items()})
    if isinstance(x, (list, tuple)):
        return tuple(freeze(v) for v in x)
    raise ContractError(f"parameter value of type {type(x).__name__} is not allowed")


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
        for name in ("table", "column", "source_version", "availability"):
            _check_type(name, getattr(self, name), str)
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
        _check_type("label column", self.column, str)
        _check_type("horizon", self.horizon, int, positive=True)
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
        _require(isinstance(self.features, (list, tuple)), "features must be a sequence of FeatureSpec")
        object.__setattr__(self, "features", tuple(self.features))      # never the caller's list
        _require(all(isinstance(f, FeatureSpec) for f in self.features), "every feature must be a FeatureSpec")
        _require(isinstance(self.label, LabelSpec), "label must be a LabelSpec")
        _check_type("universe", self.universe, str)
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
    first_date: date                 # the split covers [first_date, last_date] exactly (the dataset period)
    last_date: date
    train_sessions: int | None = None    # rolling window length
    min_train_rows: int = 1

    def __post_init__(self) -> None:
        _check_type("kind", self.kind, str)
        _require(self.train_sessions is None or isinstance(self.train_sessions, int) and not
                 isinstance(self.train_sessions, bool), "train_sessions must be an int or None")
        _require(self.kind in ("expanding", "rolling"), f"split kind {self.kind!r} is not chronological")
        _require(self.kind == "expanding" or (self.train_sessions or 0) >= self.train_min_sessions,
                 "a rolling split needs train_sessions >= train_min_sessions")
        _require(self.kind == "rolling" or self.train_sessions is None, "train_sessions is for rolling splits only")
        for name in ("train_min_sessions", "test_sessions", "step_sessions", "min_train_rows"):
            _check_type(name, getattr(self, name), int, positive=True)
        _check_type("embargo_sessions", self.embargo_sessions, int)
        _require(self.embargo_sessions >= 0, "embargo_sessions must be >= 0")
        _check_date("first_date", self.first_date)
        _check_date("last_date", self.last_date)
        _require(RESEARCH_START <= self.first_date <= self.last_date, "split dates must be ordered and in research")

    @property
    def version(self) -> str:
        return digest(self)


@dataclass(frozen=True)
class BaselineSpec:
    """The Factor Engine score the experiment must beat, pinned to the factor definitions' version."""
    factor: str
    source_version: str
    field: str = "rank_pct"

    def __post_init__(self) -> None:
        for name in ("factor", "source_version", "field"):
            _check_type(f"baseline {name}", getattr(self, name), str)
        _require(self.factor in factors.FACTORS, f"baseline {self.factor!r} is not a Factor Engine factor")
        _require(self.field in allowlist.FACTOR_FIELDS, f"baseline field {self.field!r} is not a factor score")
        current = allowlist.source_version("stock_factors")
        _require(self.source_version == current, f"baseline version {self.source_version!r} is not the current "
                                                 f"{current!r}: the factor definitions changed")

    @classmethod
    def current(cls, factor: str, field: str = "rank_pct") -> "BaselineSpec":
        return cls(factor, allowlist.source_version("stock_factors"), field)


@dataclass(frozen=True)
class ExperimentSpec:
    hypothesis: str
    dataset: DatasetSpec
    split: SplitSpec
    model: str
    seed: int
    baseline: BaselineSpec           # the Factor Engine score to beat, version-pinned
    primary_test: str
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("hypothesis", "model", "primary_test"):
            _check_type(name, getattr(self, name), str)
        _check_type("seed", self.seed, int)
        _require(isinstance(self.dataset, DatasetSpec) and isinstance(self.split, SplitSpec)
                 and isinstance(self.baseline, BaselineSpec), "dataset, split and baseline must be specs")
        _require(isinstance(self.params, Mapping), "params must be a mapping")
        _require(bool(self.hypothesis.strip()), "an experiment needs a written hypothesis")
        _require(self.model in MODELS, f"model {self.model!r} is not an approved family {MODELS}")
        _require(self.seed >= 0, "an experiment needs a seed >= 0")
        _require(self.primary_test in PRIMARY_TESTS, f"primary test {self.primary_test!r} is not declared")
        _require((self.split.first_date, self.split.last_date) == (self.dataset.period_start, self.dataset.period_end),
                 f"the split covers {self.split.first_date}..{self.split.last_date} but the dataset is "
                 f"{self.dataset.period_start}..{self.dataset.period_end}: they must be the same period")
        _require(all(isinstance(k, str) and re.fullmatch(r"[a-z_]+", k) for k in self.params),
                 "parameter names must be lower snake case")
        object.__setattr__(self, "params", freeze(self.params))         # deep, read-only copy
        digest(self.params)

    @property
    def experiment_id(self) -> str:
        return digest(self)

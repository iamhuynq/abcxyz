"""Versioned risk policies and limit evaluation (docs/risk-engine-plan.md, phase C). Pure functions.

A policy is a frozen, named and versioned set of limits; a change is a new version, never an edit. A limit
compares one metric with a warning and a breach threshold, inclusive at the threshold. A metric that could not be
computed (None) gives INCOMPLETE: missing data never passes. The overall status is the most severe state, with
INCOMPLETE first because a result without its inputs cannot be trusted.
"""

import json
from dataclasses import asdict, dataclass, field

PASS, WARNING, BREACH, INCOMPLETE = "PASS", "WARNING", "BREACH", "INCOMPLETE"
SEVERITY = {PASS: 0, WARNING: 1, BREACH: 2, INCOMPLETE: 3}


@dataclass(frozen=True)
class Limit:
    name: str
    metric: str
    direction: str                  # "max": the metric must stay below; "min": it must stay above
    warning: float
    breach: float
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.direction not in ("max", "min"):
            raise ValueError(f"direction must be max or min, not {self.direction!r}")
        if (self.direction == "max" and self.warning > self.breach) or \
                (self.direction == "min" and self.warning < self.breach):
            raise ValueError(f"limit {self.name}: the warning must come before the breach")


@dataclass(frozen=True)
class Policy:
    policy_id: str
    version: str
    limits: tuple[Limit, ...]
    provisional: bool = True
    note: str = ""

    @property
    def key(self) -> str:
        return f"{self.policy_id}/{self.version}"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass(frozen=True)
class LimitResult:
    limit: str
    metric: str
    value: float | None
    warning: float
    breach: float
    status: str
    reason: str
    affected: tuple[str, ...] = field(default_factory=tuple)


def check(limit: Limit, value: float | None, affected: tuple[str, ...] = (), missing: str = "") -> LimitResult:
    """`missing`: why the metric is unavailable (a data-quality condition); it makes the result INCOMPLETE."""
    if value is None or missing:
        why = missing or "metric not available"
        return LimitResult(limit.name, limit.metric, value, limit.warning, limit.breach, INCOMPLETE, why, affected)
    over = (lambda t: value >= t) if limit.direction == "max" else (lambda t: value <= t)
    side = ">=" if limit.direction == "max" else "<="
    if over(limit.breach):
        status, reason = BREACH, f"{value:.4g} {side} breach {limit.breach:g}"
    elif over(limit.warning):
        status, reason = WARNING, f"{value:.4g} {side} warning {limit.warning:g}"
    else:
        status, reason = PASS, ""
    return LimitResult(limit.name, limit.metric, value, limit.warning, limit.breach, status, reason, affected)


def evaluate(policy: Policy, metrics: dict[str, float | None], affected: dict[str, tuple[str, ...]] | None = None,
             missing: dict[str, str] | None = None) -> list[LimitResult]:
    """One result per limit of the policy. A metric absent from `metrics` is INCOMPLETE."""
    affected, missing = affected or {}, missing or {}
    return [check(lim, metrics.get(lim.metric), affected.get(lim.metric, ()), missing.get(lim.metric, ""))
            for lim in policy.limits]


def overall(results: list[LimitResult]) -> str:
    return max((r.status for r in results), key=SEVERITY.__getitem__, default=INCOMPLETE)


DEFAULT_V1 = Policy("default", "v1", (
    Limit("position weight", "max_position_weight", "max", 0.10, 0.15,
          "the frozen strategy holds 20 names at 5% each"),
    Limit("industry weight (ICB level 2)", "max_industry_weight", "max", 0.30, 0.40,
          "the construction grid caps industries at 30% at rebalance; actual weights reached 36% to 44%"),
    Limit("cluster weight", "max_cluster_weight", "max", 0.30, 0.40, "same logic for hidden concentration"),
    Limit("absolute beta", "abs_beta", "max", 1.2, 1.5, "a long-only beta above 1.5 is a leveraged market bet"),
    Limit("effective bets", "effective_bets", "min", 5.0, 3.0, "diversification floor"),
    Limit("illiquid weight", "illiquid_weight", "max", 0.20, 0.40,
          "positions needing more than 1 session at 10% of ADV20; the frozen strategy hits the 5% ADV cap "
          "from 10 bn VND"),
    Limit("stale-price weight", "stale_weight", "max", 0.05, 0.15,
          "positions without a trade for more than 5 sessions (the engines' STALE_SESSIONS)"),
    Limit("max drawdown (worse valuation)", "max_drawdown", "min", -0.20, -0.30, "monitoring level"),
), provisional=True, note="Provisional (2026-11-11): declared before any evaluation, not tuned on any outcome.")

POLICIES = {DEFAULT_V1.key: DEFAULT_V1}

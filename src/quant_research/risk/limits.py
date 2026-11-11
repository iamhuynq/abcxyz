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
EPS = 1e-12      # "at the threshold" up to float rounding: 1.0 / 1.25 - 1 is -0.19999999999999996, not -0.2


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
    over = (lambda t: value >= t - EPS) if limit.direction == "max" else (lambda t: value <= t + EPS)
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

# v2 (PR #8 review, 2026-11-11): the drawdown limit of v1 mixed the worst historical drawdown with a current,
# hypothetical write-down drawdown. v2 checks them separately, with the same thresholds; every other limit is v1's.
# v1 stays frozen and evaluable (its max_drawdown is the documented worse of the two).
DEFAULT_V2 = Policy("default", "v2", DEFAULT_V1.limits[:-1] + (
    Limit("max drawdown, historical (worst stored curve)", "max_drawdown_historical", "min", -0.20, -0.30,
          "monitoring level; observed peak-to-trough on the stored equity curves"),
    Limit("current drawdown with write-down", "current_drawdown_write_down", "min", -0.20, -0.30,
          "current NAV with positions stuck for more than 60 sessions at 0, against the curve's peak (valuation "
          "rule 2026-10-09); a valuation assumption, not an observed loss"),
), provisional=True, note="Provisional (2026-11-11): v1 with the drawdown split into historical and current "
                          "write-down; declared before any evaluation under it, not tuned on any outcome.")

DEFAULT = DEFAULT_V2
POLICIES = {p.key: p for p in (DEFAULT_V1, DEFAULT_V2)}

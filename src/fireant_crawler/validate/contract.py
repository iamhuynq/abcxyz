"""Payload contracts: the shape of each FireAnt response the crawler ingests.

The normalizers read fields with `raw.get(key)`, so a renamed or dropped field would become NULL without an
error. The contracts below list every field of every ingested payload with the JSON kinds seen in the whole
raw store (2026-11-11 survey, about 6.6 million rows), and whether NULL was ever seen. A response that drops
a field, changes a kind, sends a date in another format or changes the payload shape is an error; a new
field is a warning (an error for quotes, whose normalizer already rejects unknown fields).

Reports name the job, the field and the file; they never show values.
"""

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from fireant_crawler.store.raw_store import RawStore

Kind = Literal["number", "integer", "string", "bool", "null"]
Level = Literal["error", "warning"]

# Plain date or date-time, as FireAnt sends it ("2026-10-01T00:00:00"). The regex fixes the shape; the value
# must then parse with datetime.fromisoformat, as in the normalizer's `parse_date` (no 2026-02-31, no 29:70).
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2})?)?")


def is_iso_date(value: str) -> bool:
    if not _ISO_DATE.fullmatch(value):
        return False
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class Field:
    kinds: frozenset[str]
    date: bool = False       # a string that must be an ISO date or date-time


def _f(*kinds: str, date: bool = False) -> Field:
    return Field(frozenset(kinds), date)


NUM, NUM_NULL, STR, STR_NULL = _f("number"), _f("number", "null"), _f("string"), _f("string", "null")
DATE, DATE_NULL = _f("string", date=True), _f("string", "null", date=True)


@dataclass(frozen=True)
class Contract:
    payload: Literal["list", "object"]
    fields: dict[str, Field]
    unknown_level: Level = "warning"


_QUOTE_NUMBERS = ("priceOpen", "priceHigh", "priceLow", "priceClose", "priceAverage", "priceBasic",
                  "totalVolume", "dealVolume", "putthroughVolume", "totalValue", "putthroughValue",
                  "buyForeignQuantity", "buyForeignValue", "sellForeignQuantity", "sellForeignValue",
                  "buyCount", "buyQuantity", "sellCount", "sellQuantity", "adjRatio", "currentForeignRoom", "unit")
_QUOTE_NULLABLE = ("propTradingNetDealValue", "propTradingNetPTValue", "propTradingNetValue")
_FUNDAMENTAL_NUMBERS = ("sharesOutstanding", "freeShares", "beta", "dividend", "dividendYield", "marketCap",
                        "low52Week", "high52Week", "priceChange1y", "avgVolume10d", "avgVolume3m", "pe", "eps",
                        "sales_TTM", "netProfit_TTM", "insiderOwnership", "institutionOwnership",
                        "foreignOwnership")
_SYMBOL_INFO = {"symbol": STR, "name": STR, "exchange": STR_NULL, "type": STR, "isListing": _f("bool"),
                "industryCode": STR_NULL, "icbCode": STR_NULL, "instrument": _f("string", "null")}

CONTRACTS: dict[str, Contract] = {
    "quotes": Contract("list", {"date": DATE, "symbol": STR, **{k: NUM for k in _QUOTE_NUMBERS},
                                **{k: NUM_NULL for k in _QUOTE_NULLABLE}}, unknown_level="error"),
    "corporate_actions": Contract("list", {
        "eventID": _f("integer"), "symbol": STR, "name": STR, "title": STR, "type": _f("integer"),
        "recordDate": DATE, "registrationDate": DATE, "executionDate": DATE_NULL}),
    "report_marks": Contract("list", {"id": STR, "label": STR, "date": DATE, "title": STR, "color": STR}),
    "fundamental": Contract("object", {"symbol": STR_NULL, "companyType": _f("integer"),
                                       **{k: NUM_NULL for k in _FUNDAMENTAL_NUMBERS}}),
    "icb": Contract("list", {"industryCode": STR, "level": _f("integer"), "name": STR, "description": STR}),
    "universe_search": Contract("list", _SYMBOL_INFO),
    "universe_symbol": Contract("object", _SYMBOL_INFO),
}
CHECKED_JOBS = ("quotes", "corporate_actions", "report_marks", "fundamental", "icb", "universe")


def contract_name(job: str, path: str) -> str | None:
    """The contract for a raw response, or None for a response kept raw only (`/instruments`)."""
    if job != "universe":
        return job if job in CONTRACTS else None
    if path == "/symbols/search":
        return "universe_search"
    if path.startswith("/symbols/"):
        return "universe_symbol"
    return None


def kind_of(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def _allowed(kind: str, spec: Field) -> bool:
    return kind in spec.kinds or (kind == "integer" and "number" in spec.kinds)


@dataclass(frozen=True)
class Violation:
    level: Level
    contract: str
    rule: str          # missing_field, wrong_kind, bad_date, unknown_field, wrong_payload
    field: str


def check_payload(name: str, data: Any) -> list[Violation]:
    """Violations of one payload against its contract, one per (rule, field)."""
    contract = CONTRACTS[name]
    expected = list if contract.payload == "list" else dict
    if not isinstance(data, expected):
        return [Violation("error", name, "wrong_payload", contract.payload)]
    rows = data if isinstance(data, list) else [data]
    found: set[Violation] = set()
    for row in rows:
        if not isinstance(row, dict):
            found.add(Violation("error", name, "wrong_payload", "row"))
            continue
        for key, spec in contract.fields.items():
            if key not in row:
                found.add(Violation("error", name, "missing_field", key))
                continue
            value = row[key]
            if not _allowed(kind_of(value), spec):
                found.add(Violation("error", name, "wrong_kind", key))
            elif spec.date and isinstance(value, str) and not is_iso_date(value):
                found.add(Violation("error", name, "bad_date", key))
        for key in row.keys() - contract.fields.keys():
            found.add(Violation(contract.unknown_level, name, "unknown_field", str(key)))
    return sorted(found, key=lambda v: (v.level, v.rule, v.field))


@dataclass
class ContractResult:
    files: int = 0
    skipped: int = 0                                   # raw-only responses (no contract)
    counts: Counter = field(default_factory=Counter)   # Violation -> number of files
    examples: dict = field(default_factory=dict)       # Violation -> first file

    @property
    def errors(self) -> int:
        return sum(n for v, n in self.counts.items() if v.level == "error")

    def add(self, path: Path, violations: Iterable[Violation]) -> None:
        self.files += 1
        for v in violations:
            self.counts[v] += 1
            self.examples.setdefault(v, path)


def check_files(files: Iterable[tuple[str, Path]], result: ContractResult | None = None) -> ContractResult:
    """Check (job, path) raw files. Files that cannot be read count as a wrong payload."""
    result = result or ContractResult()
    for job, path in files:
        try:
            envelope = RawStore.read(path)
            name = contract_name(job, envelope["request"]["path"])
            data = envelope["data"]
        except (OSError, ValueError, KeyError, TypeError):
            result.add(path, [Violation("error", job, "wrong_payload", "envelope")])
            continue
        if name is None:
            result.skipped += 1
            continue
        result.add(path, check_payload(name, data))
    return result


def raw_files(store: RawStore, jobs: Iterable[str] = CHECKED_JOBS, since: str | None = None,
              latest_only: bool = False) -> list[tuple[str, Path]]:
    """(job, path) of stored raw files: all of them, those fetched on or after `since`, or only those of each
    job's latest fetch date."""
    out: list[tuple[str, Path]] = []
    for job in jobs:
        paths = [p for _, p in store.iter_files(job)]
        if latest_only and paths:
            latest = max(p.parts[-2] for p in paths)
            paths = [p for p in paths if p.parts[-2] == latest]
        if since:
            paths = [p for p in paths if p.parts[-2] >= since]
        out += [(job, p) for p in paths]
    return out


def render_contract_section(result: ContractResult, scope: str) -> list[str]:
    lines = ["", f"## Contract ({scope})", "",
             f"{result.files:,} raw files checked ({result.skipped:,} raw-only responses skipped); "
             f"{result.errors:,} error and {sum(result.counts.values()) - result.errors:,} warning findings."]
    if result.counts:
        lines += ["", "| level | contract | rule | field | files | example file |", "|---|---|---|---|---|---|"]
        for v in sorted(result.counts, key=lambda v: (v.level, v.contract, v.rule, v.field)):
            example = "/".join(result.examples[v].parts[-4:])
            lines.append(f"| {v.level} | {v.contract} | {v.rule} | {v.field} | {result.counts[v]:,} | {example} |")
    return lines

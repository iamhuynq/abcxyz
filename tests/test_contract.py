"""Payload contracts (validate/contract.py): fixtures pass, normalizers read only contracted fields, every
rule fires, and a broken raw file stops `fireant validate` unless --no-contract is given."""

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fireant_crawler import cli
from fireant_crawler.client.fireant_client import ApiResponse
from fireant_crawler.normalize.corporate_actions import normalize_corporate_actions
from fireant_crawler.normalize.quotes import normalize_quotes
from fireant_crawler.normalize.reference import normalize_fundamental, normalize_icb, normalize_symbol
from fireant_crawler.normalize.report_marks import normalize_report_marks
from fireant_crawler.store.raw_store import RawStore
from fireant_crawler.validate.contract import (CONTRACTS, Violation, check_files, check_payload,
                                               contract_name, raw_files)
from tests.research.synthetic import build_synthetic_warehouse

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 11, 11, 9, 0, tzinfo=UTC)

# fixture file -> contract (fixtures are responses of the real API, trimmed)
FIXTURE_CONTRACTS = {
    "quotes_vos_2026_07.json": "quotes", "quotes_vos_2026_08.json": "quotes", "quotes_flc_tail.json": "quotes",
    "quotes_vnindex_sample.json": "quotes", "quotes_vos_adj_points.json": "quotes",
    "events_sample.json": "corporate_actions", "timescale_marks_vos.json": "report_marks",
    "fundamental_vos.json": "fundamental", "icb_sample.json": "icb",
    "symbol_flc.json": "universe_symbol", "symbol_vnindex.json": "universe_symbol",
}


def _load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(FIXTURE_CONTRACTS))
def test_fixtures_satisfy_their_contract(name):
    assert check_payload(FIXTURE_CONTRACTS[name], _load(name)) == []


class _Recorder(dict):
    """A payload row that records every key a normalizer reads."""
    def __init__(self, data, seen):
        super().__init__(data)
        self.seen = seen

    def get(self, key, default=None):
        self.seen.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.seen.add(key)
        return super().__getitem__(key)


def _keys_read(normalize, contract, sample):
    seen: set[str] = set()
    rows = sample if isinstance(sample, list) else [sample]
    wrapped = [_Recorder(r, seen) for r in rows]
    normalize(wrapped if isinstance(sample, list) else wrapped[0])
    assert seen, "the recorder saw no reads"
    return seen - set(CONTRACTS[contract].fields)


def test_normalizers_read_only_contracted_fields():
    """Contract and code cannot drift apart: every key a normalizer reads is declared in its contract."""
    cases = [
        (lambda r: normalize_quotes("VOS", r, NOW), "quotes", _load("quotes_vos_2026_07.json")),
        (lambda r: normalize_corporate_actions(r, NOW), "corporate_actions", _load("events_sample.json")),
        (lambda r: normalize_report_marks("VOS", r, NOW), "report_marks", _load("timescale_marks_vos.json")),
        (lambda r: normalize_fundamental("VOS", r, NOW), "fundamental", _load("fundamental_vos.json")),
        (lambda r: normalize_icb(r, NOW), "icb", _load("icb_sample.json")),
        (lambda r: normalize_symbol(r, NOW, source="symbol"), "universe_symbol", _load("symbol_flc.json")),
        (lambda r: normalize_symbol(r, NOW, source="search"), "universe_search", _load("symbol_flc.json")),
    ]
    for normalize, contract, sample in cases:
        assert _keys_read(normalize, contract, sample) == set(), contract


def _quote():
    return copy.deepcopy(_load("quotes_vos_2026_07.json")[:3])


def test_missing_field_is_an_error():
    rows = _quote()
    del rows[1]["priceClose"]
    assert check_payload("quotes", rows) == [Violation("error", "quotes", "missing_field", "priceClose")]


def test_null_is_an_error_only_where_never_seen():
    rows = _quote()
    rows[0]["priceClose"] = None
    rows[0]["propTradingNetValue"] = None
    assert check_payload("quotes", rows) == [Violation("error", "quotes", "wrong_kind", "priceClose")]


def test_wrong_kind_is_an_error():
    rows = _quote()
    rows[2]["totalVolume"] = "1.000"
    assert check_payload("quotes", rows) == [Violation("error", "quotes", "wrong_kind", "totalVolume")]


def test_integer_fields_reject_floats_and_number_fields_accept_integers():
    events = copy.deepcopy(_load("events_sample.json")[:2])
    events[0]["eventID"] = float(events[0]["eventID"])
    assert check_payload("corporate_actions", events) == [
        Violation("error", "corporate_actions", "wrong_kind", "eventID")]
    rows = _quote()
    rows[0]["priceClose"] = 25
    assert check_payload("quotes", rows) == []


def test_bad_date_is_an_error():
    marks = copy.deepcopy(_load("timescale_marks_vos.json")[:2])
    marks[1]["date"] = "11/11/2026"
    assert check_payload("report_marks", marks) == [Violation("error", "report_marks", "bad_date", "date")]


@pytest.mark.parametrize("value", ["2026-02-31", "2026-13-01", "2026-99-99", "2026-11-11T29:00:00",
                                   "2026-11-11T10:70:00", "2026-11-11T10:00:61", "2026-11-11T00:00:00+25:00",
                                   "2026-11-11 00:00:00", "2026-11-11T00:00:00+0700"])
def test_impossible_or_malformed_dates_are_errors(value):
    marks = copy.deepcopy(_load("timescale_marks_vos.json")[:1])
    marks[0]["date"] = value
    assert check_payload("report_marks", marks) == [Violation("error", "report_marks", "bad_date", "date")]


@pytest.mark.parametrize("value", ["2026-11-11", "2026-11-11T00:00:00", "2026-11-11T09:15:30.123",
                                   "2026-11-11T00:00:00Z", "2026-11-11T00:00:00+07:00", "2028-02-29T00:00:00"])
def test_valid_dates_pass(value):
    marks = copy.deepcopy(_load("timescale_marks_vos.json")[:1])
    marks[0]["date"] = value
    assert check_payload("report_marks", marks) == []


def test_unknown_field_is_a_warning_except_for_quotes():
    marks = copy.deepcopy(_load("timescale_marks_vos.json")[:1])
    marks[0]["releaseDate"] = "2026-11-11"
    assert check_payload("report_marks", marks) == [
        Violation("warning", "report_marks", "unknown_field", "releaseDate")]
    rows = _quote()
    rows[0]["priceCeiling"] = 1.0
    assert check_payload("quotes", rows) == [Violation("error", "quotes", "unknown_field", "priceCeiling")]


def test_wrong_payload_shape_is_an_error():
    assert check_payload("fundamental", [_load("fundamental_vos.json")]) == [
        Violation("error", "fundamental", "wrong_payload", "object")]
    assert check_payload("icb", {"data": []}) == [Violation("error", "icb", "wrong_payload", "list")]
    assert check_payload("icb", ["x"]) == [Violation("error", "icb", "wrong_payload", "row")]


def test_contract_names():
    assert contract_name("universe", "/symbols/search") == "universe_search"
    assert contract_name("universe", "/symbols/VOS") == "universe_symbol"
    assert contract_name("universe", "/instruments") is None
    assert contract_name("quotes", "/symbols/VOS/historical-quotes") == "quotes"


def _write(store: RawStore, job: str, key: str, path: str, data, at: datetime) -> Path:
    return store.write(job, key, "c", ApiResponse(path, {}, 200, data, {}, 0.0, 1), at)


def test_latest_only_checks_the_latest_fetch_date_of_each_job(tmp_path):
    store = RawStore(tmp_path)
    old_bad = _quote()
    del old_bad[0]["date"]
    _write(store, "quotes", "VOS", "/symbols/VOS/historical-quotes", old_bad, datetime(2026, 11, 9, tzinfo=UTC))
    _write(store, "quotes", "VOS", "/symbols/VOS/historical-quotes", _quote(), NOW)
    _write(store, "universe", "ALL", "/instruments", [{"symbol": "VOS"}], NOW)
    assert check_files(raw_files(store, latest_only=True)).errors == 0
    full = check_files(raw_files(store))
    assert (full.files, full.skipped, full.errors) == (2, 1, 1)
    assert check_files(raw_files(store, since="2026-11-10")).errors == 0


def test_unreadable_file_is_an_error(tmp_path):
    bad = tmp_path / "raw" / "quotes" / "VOS" / "2026-11-11" / "c__090000.json.gz"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not gzip")
    assert check_files(raw_files(RawStore(tmp_path))).errors == 1


def test_validate_stops_on_a_contract_error_unless_skipped(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    data.mkdir()
    build_synthetic_warehouse(data / "warehouse.duckdb")
    env_file = tmp_path / ".env"
    env_file.write_text("FIREANT_TOKEN=\n")
    monkeypatch.delenv("FIREANT_TOKEN", raising=False)
    monkeypatch.setenv("DATA_DIR", str(data))
    monkeypatch.setenv("ENV_FILE", str(env_file))
    store = RawStore(data)
    _write(store, "quotes", "VOS", "/symbols/VOS/historical-quotes", _quote(), NOW)
    assert cli.main(["validate"]) == 0
    renamed = _quote()
    for row in renamed:
        row["closePrice"] = row.pop("priceClose")
    _write(store, "quotes", "VOS", "/symbols/VOS/historical-quotes", renamed,
           datetime(2026, 11, 12, 9, 0, tzinfo=UTC))
    assert cli.main(["validate"]) == 1
    report = next((data / "reports").glob("validation-*.md")).read_text()
    assert "| error | quotes | missing_field | priceClose | 1 |" in report
    assert cli.main(["validate", "--no-contract"]) == 0
    assert cli.main(["contract", "scan"]) == 1
    assert cli.main(["contract", "scan", "--since", "2026-11-12", "--job", "report_marks"]) == 0
    capsys.readouterr()

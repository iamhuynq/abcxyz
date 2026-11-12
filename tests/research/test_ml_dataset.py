"""ML dataset builder (docs/ml-plan.md, PR #11): provenance first, exact per-symbol label intervals, one status per
row in a declared order, reconciled accounting, canonical fingerprint, atomic read-only artifacts."""

import hashlib
import os
import subprocess
import sys
from datetime import UTC, date, datetime

import duckdb
import pandas as pd
import pytest

from quant_research.build import build
from quant_research.factors import FactorParams
from quant_research.ml import allowlist, dataset
from quant_research.ml.contracts import ContractError, DatasetSpec, FeatureSpec, LabelSpec
from tests.research.synthetic import build_synthetic_warehouse

NOW = datetime(2026, 11, 12, 9, 0, tzinfo=UTC)
START = date(2022, 1, 3)
F = FeatureSpec.current
NO_TRADE = __import__("tests.research.synthetic", fromlist=["calendar"]).calendar(START)[150]


def _spec(h: int = 5, features=None, start=date(2022, 1, 3), end=date(2023, 12, 29)) -> DatasetSpec:
    feats = features or (F("stock_features", "return_5d"), F("stock_features", "volume_ratio_20"))
    return DatasetSpec(tuple(feats), LabelSpec(f"fwd_ret_exec_{h}d", h), start, end)


@pytest.fixture(scope="module")
def research(tmp_path_factory):
    d = tmp_path_factory.mktemp("mlds")
    build_synthetic_warehouse(d / "w.duckdb", start=START)
    with duckdb.connect(str(d / "w.duckdb")) as w:          # one no-trade session inside UPC's series
        w.execute("""UPDATE quotes_daily SET deal_volume = 0, total_volume = 0, total_value = 0, putthrough_value = 0
                     WHERE symbol = 'UPC' AND date = ?""", [NO_TRADE])
    build(d / "w.duckdb", d / "r.duckdb", NOW)
    return d / "r.duckdb"


def _con(path):
    return duckdb.connect(str(path), read_only=True)


def _build_id(con):
    return con.execute("SELECT build_id FROM feature_builds ORDER BY built_at DESC LIMIT 1").fetchone()[0]


def test_dataset_is_built_read_only_and_accounting_reconciles(research):
    before = hashlib.sha256(research.read_bytes()).hexdigest()
    with _con(research) as con:
        r = dataset.build_dataset(con, _spec(), _build_id(con))
    assert hashlib.sha256(research.read_bytes()).hexdigest() == before          # nothing written
    assert [e[0] for e in r.exclusions] == list(dataset.REASONS)              # every reason, zeros included
    assert r.input_rows == r.kept_rows + r.excluded_rows == r.kept_rows + sum(c for _r, c, _s in r.exclusions)
    assert all(s == pytest.approx(c / r.input_rows) for _r, c, s in r.exclusions)
    assert r.kept_rows > 50 and dict((e[0], e[1]) for e in r.exclusions)["outside_universe"] > 0
    assert list(r.data.columns) == ["date", "symbol", "stock_features.return_5d", "stock_features.volume_ratio_20",
                                    "fwd_ret_exec_5d", "label_start", "label_end"]
    assert not r.data.drop(columns=[]).isna().any().any()
    assert r.metadata["exclusions"] == [list(e) for e in r.exclusions] and r.metadata["kept_rows"] == r.kept_rows
    assert "z_industry excluded" in r.metadata["limitations"][0] and r.metadata["warehouse_sha"]


@pytest.mark.parametrize("h", [3, 5, 10, 20])
def test_intervals_reproduce_the_label_from_the_full_symbol_series(research, h):
    with _con(research) as con:
        full = dataset.build_dataset(con, _spec(h), _build_id(con))
        late = dataset.build_dataset(con, _spec(h, start=date(2022, 9, 1)), _build_id(con))   # starts mid-series
        px = con.execute("SELECT symbol, date, adj_open, adj_close FROM daily_panel").df()
    px["date"] = pd.to_datetime(px["date"]).dt.date
    o = px.set_index(["symbol", "date"])["adj_open"]
    c = px.set_index(["symbol", "date"])["adj_close"]
    for r in full.data.itertuples(index=False):
        assert r.date < r.label_start <= r.label_end
        assert c[(r.symbol, r.label_end)] / o[(r.symbol, r.label_start)] - 1 == pytest.approx(r[4], abs=1e-12)
        dates = sorted(px.loc[px["symbol"] == r.symbol, "date"])                 # the symbol's own sessions
        i = dates.index(r.date)
        assert (r.label_start, r.label_end) == (dates[i + 1], dates[i + h])
    merged = late.data.merge(full.data, on=["date", "symbol"], suffixes=("", "_full"))
    assert len(merged) == len(late.data) and (merged["label_end"] == merged["label_end_full"]).all()


def test_masks_are_never_hidden_by_a_null_label_and_first_match_wins(research):
    with _con(research) as con:
        bid = _build_id(con)
        spec = _spec(20)
        df = dataset._load(con, spec, "")
        status = dataset._status(df, spec)
    crossing = df["label_end"].notna() & (df["label_end_period"] != df["decision_period"])
    masked = df["entry_blocked"].eq(True) | df["price_jump_in_window"].eq(True) | crossing
    assert not (masked & df["label"].isna() & df["label_end"].notna()).any()    # masks never null the label
    assert (status[df["label_end"].isna() & df["in_universe"]] == "no_label_interval").all()
    assert set(status) <= {"kept", *dataset.REASONS}
    assert (status[~df["in_universe"]] == "outside_universe").all()            # first in the order
    assert bid


def test_null_features_are_excluded_and_counted_never_filled(research):
    feats = (F("stock_features", "return_5d"), F("stock_factors", "order_flow.rank_pct"))  # factors: < 30 stocks
    with _con(research) as con:
        r = dataset.build_dataset(con, _spec(features=feats), _build_id(con))
    counts = dict((e[0], e[1]) for e in r.exclusions)
    assert counts["feature_null"] > 0 and r.kept_rows == 0                      # no factor score in a 3-stock market


def test_provenance_is_checked_before_reading(research, tmp_path):
    path = tmp_path / "r.duckdb"
    path.write_bytes(research.read_bytes())
    with duckdb.connect(str(path)) as rw:
        bid = _build_id(rw)
        rw.execute("UPDATE feature_builds SET warehouse_sha = ''")
    with _con(path) as con:
        with pytest.raises(ContractError, match="no warehouse_sha"):
            dataset.build_dataset(con, _spec(), bid)
    with duckdb.connect(str(path)) as rw:
        rw.execute("UPDATE feature_builds SET factor_params = ?",
                   [__import__("json").dumps(__import__("dataclasses").asdict(FactorParams(min_stocks=1)))])
    with _con(path) as con:
        with pytest.raises(ContractError, match="not produced under the approved configuration"):
            dataset.build_dataset(con, _spec(features=(F("stock_factors", "order_flow.rank_pct"),)), bid)
        with pytest.raises(ContractError, match="no feature build"):
            dataset.build_dataset(con, _spec(), "missing")


def test_z_industry_and_targets_cannot_be_features():
    assert not any(c.endswith("z_industry") for c in allowlist.STOCK_FACTORS) and len(allowlist.STOCK_FACTORS) == 18
    for table, column in (("stock_factors", "order_flow.z_industry"), ("stock_targets", "fwd_ret_exec_5d")):
        with pytest.raises(ContractError):
            FeatureSpec(table, column, "x")


def test_fingerprint_covers_content_and_provenance(research):
    with _con(research) as con:
        bid = _build_id(con)
        a = dataset.build_dataset(con, _spec(), bid)
        b = dataset.build_dataset(con, _spec(), bid)
        other_order = dataset.build_dataset(con, _spec(features=(F("stock_features", "volume_ratio_20"),
                                                                 F("stock_features", "return_5d"))), bid)
        other_label = dataset.build_dataset(con, _spec(10), bid)
    assert a.fingerprint == b.fingerprint
    assert len({a.fingerprint, other_order.fingerprint, other_label.fingerprint}) == 3
    changed = a.data.copy()
    changed.iloc[0, 2] = changed.iloc[0, 2] + 1e-12
    assert dataset.content_hash(changed) != a.metadata["content_hash"]
    meta = dict(a.metadata) | {"warehouse_sha": "0" * 64}
    assert dataset.digest(meta) != a.fingerprint


def test_artifact_is_atomic_verified_and_never_overwritten(research, tmp_path, monkeypatch):
    with _con(research) as con:
        r = dataset.build_dataset(con, _spec(), _build_id(con))
    out = dataset.write_dataset(r, tmp_path / "ds")
    data, doc = dataset.read_dataset(out)
    assert doc["fingerprint"] == r.fingerprint and len(data) == r.kept_rows
    with pytest.raises(FileExistsError):
        dataset.write_dataset(r, out)
    import json as _json
    real_dumps = _json.dumps

    def failing(*a, **k):                                   # fail while writing the metadata
        if k.get("indent") == 1:
            raise OSError("disk full")
        return real_dumps(*a, **k)
    monkeypatch.setattr(dataset.json, "dumps", failing)
    with pytest.raises(OSError):
        dataset.write_dataset(r, tmp_path / "ds2")
    monkeypatch.undo()
    assert not (tmp_path / "ds2").exists() and not list(tmp_path.glob(".ds2.tmp-*"))
    monkeypatch.setattr(type(tmp_path), "rename", lambda self, target: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError):
        dataset.write_dataset(r, tmp_path / "ds3")
    monkeypatch.undo()
    assert not (tmp_path / "ds3").exists() and not list(tmp_path.glob(".ds3.tmp-*"))
    tampered = pd.read_parquet(out / "data.parquet")
    tampered.iloc[0, 2] = 0.123
    tampered.to_parquet(out / "data.parquet", index=False)
    with pytest.raises(ContractError, match="content hash"):
        dataset.read_dataset(out)


def test_fingerprint_is_stable_across_processes(research):
    code = ("import duckdb, sys; from tests.research.test_ml_dataset import _spec, _build_id; "
            "from quant_research.ml import dataset; con = duckdb.connect(sys.argv[1], read_only=True); "
            "print(dataset.build_dataset(con, _spec(), _build_id(con)).fingerprint)")
    outs = {subprocess.run([sys.executable, "-c", code, str(research)], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": s}).stdout.strip() for s in ("1", "999")}
    assert len(outs) == 1


def test_a_no_trade_session_is_part_of_the_interval(research):
    with _con(research) as con:
        traded = con.execute("SELECT is_traded FROM daily_panel WHERE symbol = 'UPC' AND date = ?",
                             [NO_TRADE]).fetchone()
        r = dataset.build_dataset(con, _spec(5), _build_id(con))
    assert traded == (False,)                                    # kept in the panel as a no-trade session
    spans = r.data[(r.data["symbol"] == "UPC") & (r.data["date"] < NO_TRADE) & (r.data["label_end"] >= NO_TRADE)]
    assert len(spans) > 0                                          # intervals that run over the no-trade session


def test_labels_ending_after_the_dataset_end_are_excluded(research):
    with _con(research) as con:
        r = dataset.build_dataset(con, _spec(20, end=date(2022, 12, 30)), _build_id(con))   # data ends 2023-02
    counts = dict((e[0], e[1]) for e in r.exclusions)
    assert counts["label_after_dataset_end"] > 0 and (r.data["label_end"] <= date(2022, 12, 30)).all()
    assert counts["label_crosses_period"] == 0


def test_tampered_metadata_is_refused(research, tmp_path):
    import json
    with _con(research) as con:
        r = dataset.build_dataset(con, _spec(), _build_id(con))
    out = dataset.write_dataset(r, tmp_path / "ds")
    doc = json.loads((out / "metadata.json").read_text())
    doc["metadata"]["warehouse_sha"] = "0" * 64
    (out / "metadata.json").write_text(json.dumps(doc))
    with pytest.raises(ContractError, match="does not match its fingerprint"):
        dataset.read_dataset(out)


# --- PR #11 review fixes -------------------------------------------------------------------------------------------

def _artifact(research, tmp_path, name="ds"):
    import json
    with _con(research) as con:
        r = dataset.build_dataset(con, _spec(), _build_id(con))
    out = dataset.write_dataset(r, tmp_path / name)
    return r, out, json.loads((out / "metadata.json").read_text())


def test_fingerprint_binds_provenance_and_logical_content(research, tmp_path):
    import json
    r, out, doc = _artifact(research, tmp_path)
    meta = dict(r.metadata)
    assert r.fingerprint == dataset.fingerprint(meta, dataset.content_hash(r.data)) == doc["fingerprint"]
    data = pd.read_parquet(out / "data.parquet")
    changed = data.copy()
    changed.iloc[0, 2] = changed.iloc[0, 2] + 1.0
    new_hash = dataset.content_hash(changed)
    # data and its content_hash rewritten together, fingerprint left: refused, because the fingerprint covers content
    changed.to_parquet(out / "data.parquet", index=False)
    (out / "metadata.json").write_text(json.dumps(doc | {"metadata": doc["metadata"] | {"content_hash": new_hash}}))
    with pytest.raises(ContractError, match="does not match its fingerprint"):
        dataset.read_dataset(out)


@pytest.mark.parametrize("tamper, why", [
    (lambda d: d | {"metadata": d["metadata"] | {"content_hash": "0" * 64}}, "content hash"),
    (lambda d: d | {"metadata": d["metadata"] | {"kept_rows": d["metadata"]["kept_rows"] + 1}}, "fingerprint"),
    (lambda d: d | {"fingerprint": "0" * 16}, "fingerprint"),
])
def test_tampered_metadata_or_fingerprint_is_refused(research, tmp_path, tamper, why):
    import json
    _r, out, doc = _artifact(research, tmp_path)
    (out / "metadata.json").write_text(json.dumps(tamper(doc)))
    with pytest.raises(ContractError, match=why):
        dataset.read_dataset(out)


@pytest.fixture(scope="module")
def year_end(tmp_path_factory):
    """A build whose 300 sessions run from 2023-06-01 into 2024: labels cross the research cutoff."""
    d = tmp_path_factory.mktemp("yearend")
    build_synthetic_warehouse(d / "w.duckdb", start=date(2023, 6, 1))
    build(d / "w.duckdb", d / "r.duckdb", NOW)
    return d / "r.duckdb"


def test_label_crossing_the_research_cutoff_is_checked_from_the_interval(year_end):
    spec = _spec(5, start=date(2023, 6, 1), end=date(2023, 12, 29))
    with _con(year_end) as con:
        df = dataset._load(con, spec, "")
        r = dataset.build_dataset(con, spec, _build_id(con))
    status = dataset._status(df, spec)
    end = pd.to_datetime(df["label_end"]).dt.date
    assert (status[end > date(2023, 12, 31)].isin(["outside_universe", "label_crosses_period"])).all()
    assert (status[(end > date(2023, 12, 31)) & df["in_universe"]] == "label_crosses_period").all()
    last = df[end == date(2023, 12, 29)]                                   # ends on the last research session
    assert len(last) and not (dataset._status(df, spec)[last.index] == "label_crosses_period").any()
    assert (r.data["label_end"] <= date(2023, 12, 31)).all() and dict(
        (e[0], e[1]) for e in r.exclusions)["label_crosses_period"] > 0
    assert (df["decision_period"] == "research").all()


def test_missing_feature_rows_are_counted_and_late_features_refused(research, monkeypatch):
    real = dataset._load
    with _con(research) as con:
        bid = _build_id(con)

        def late(*a):
            df = real(*a)
            return df.assign(feature_date=df["feature_date"].where(df.index != 3, pd.Timestamp("2099-01-01")))
        monkeypatch.setattr(dataset, "_load", late)
        with pytest.raises(ContractError, match="after their decision date"):
            dataset.build_dataset(con, _spec(), bid)

        def missing(*a):
            df = real(*a)
            return df.assign(feature_date=df["feature_date"].where(df.index != 3, None))
        monkeypatch.setattr(dataset, "_load", missing)
        r = dataset.build_dataset(con, _spec(), bid)
    assert r.metadata["rows_without_stock_features"] == 1


def test_canonical_cells():
    import numpy as np
    c = dataset._cell
    assert c(np.int64(3)) == c(3) == 3 and c(np.float32(0.5)) == c(0.5) == (0.5).hex()
    assert c(float("nan")) is None and c(None) is None and c(pd.NaT) is None and c(np.datetime64("NaT", "ns")) is None
    assert c(float("inf")) == "inf" and c(float("-inf")) == "-inf"
    assert c(-0.0) == c(0.0) and c(np.bool_(True)) is True and c(False) is False
    assert c(pd.Timestamp("2023-01-02")) == c(date(2023, 1, 2)) == c(np.datetime64("2023-01-02")) == "2023-01-02"
    assert c(datetime(2023, 1, 2, 9, 30)) == "2023-01-02T09:30:00"
    with pytest.raises(ContractError):
        c(object())


def test_concurrent_publishers_never_overwrite(research, tmp_path):
    import threading
    with _con(research) as con:
        r = dataset.build_dataset(con, _spec(), _build_id(con))
    target, results = tmp_path / "shared", []

    def publish():
        try:
            dataset.write_dataset(r, target)
            results.append("ok")
        except FileExistsError:
            results.append("refused")
    threads = [threading.Thread(target=publish) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == ["ok", "refused", "refused", "refused"]
    assert dataset.read_dataset(target)[1]["fingerprint"] == r.fingerprint
    assert not list(tmp_path.glob(".shared.*"))                            # no lock or temp left
    (tmp_path / ".other.lock").write_text("")                              # unreadable: never removed
    with pytest.raises(FileExistsError, match="being published"):
        dataset.write_dataset(r, tmp_path / "other")
    assert (tmp_path / ".other.lock").exists()


def test_excluded_candidates_get_intervals_from_the_full_series(research):
    with _con(research) as con:
        spec = _spec(5)
        df = dataset._load(con, spec, "")
        px = con.execute("SELECT symbol, date FROM daily_panel ORDER BY symbol, date").df()
    status = dataset._status(df, spec)
    px["date"] = pd.to_datetime(px["date"]).dt.date
    own = {s: list(g["date"]) for s, g in px.groupby("symbol")}
    excluded = df[(status != "kept") & df["label_end"].notna()]
    assert len(excluded) > 0 and set(status[excluded.index]) >= {"outside_universe"}
    for r in excluded.itertuples(index=False):
        d = pd.Timestamp(r.date).date()
        seq = own[r.symbol]
        i = seq.index(d)
        assert (pd.Timestamp(r.label_start).date(), pd.Timestamp(r.label_end).date()) == (seq[i + 1], seq[i + 5])


def test_each_crossing_rule_works_on_its_own():
    """Period change and research cutoff are two independent rules of label_crosses_period."""
    spec = _spec(5)
    base = dict(symbol="A", in_universe=True, label_start=date(2023, 12, 20), entry_blocked=False,
                price_jump_in_window=False, label=0.01, **{f.name: 1.0 for f in spec.features})
    df = pd.DataFrame([
        base | dict(date=date(2023, 12, 18), label_end=date(2023, 12, 22), decision_period="research",
                    label_end_period="validation"),                    # period differs, before the cutoff
        base | dict(date=date(2023, 12, 19), label_end=date(2024, 1, 2), decision_period="research",
                    label_end_period="research"),                      # same period label, after the cutoff
        base | dict(date=date(2023, 12, 20), label_end=date(2023, 12, 22), decision_period="research",
                    label_end_period="research")])                     # neither: kept
    assert dataset._status(df, spec).tolist() == ["label_crosses_period", "label_crosses_period", "kept"]


def test_a_lock_left_by_a_crashed_publisher_is_recovered_but_a_live_one_is_not(research, tmp_path):
    import json
    import socket
    with _con(research) as con:
        r = dataset.build_dataset(con, _spec(), _build_id(con))
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()                                                          # its pid no longer runs
    lock = tmp_path / ".ds.lock"
    owner = {"host": socket.gethostname(), "created": 0, "token": "x"}
    lock.write_text(json.dumps(owner | {"pid": dead.pid}))
    dataset.write_dataset(r, tmp_path / "ds")                            # stale: recovered
    assert (tmp_path / "ds").exists() and not lock.exists() and not list(tmp_path.glob(".ds.lock.stale-*"))
    for live in (owner | {"pid": os.getpid()}, owner | {"pid": dead.pid, "host": "another-host"}):
        lock2 = tmp_path / ".ds2.lock"
        lock2.write_text(json.dumps(live))                               # alive, or not checkable: kept
        with pytest.raises(FileExistsError, match="delete the lock file"):
            dataset.write_dataset(r, tmp_path / "ds2")
        assert json.loads(lock2.read_text()) == live
        lock2.unlink()

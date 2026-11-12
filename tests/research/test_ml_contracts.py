"""ML contracts: allowlist, label and dataset rules, research cutoff, deterministic hashes (docs/ml-plan.md)."""

import os
import subprocess
import sys
from dataclasses import replace
from datetime import date, datetime

import pandas as pd
import pytest

from quant_research import factors, regimes
from quant_research.build import FEATURE_SET_VERSION
from quant_research.ml import allowlist
from quant_research.ml.canonical import CONTRACT, HashError, canonical, digest
from quant_research.ml.contracts import (RESEARCH_END, ContractError, DatasetSpec, ExperimentSpec, FeatureSpec,
                                         LabelSpec, SplitSpec)
from quant_research.ml.keys import check_point_in_time, check_unique

F = FeatureSpec.current


def _dataset(**over) -> DatasetSpec:
    base = dict(features=(F("stock_features", "return_5d"), F("stock_factors", "order_flow.rank_pct"),
                          F("market_regimes", "direction")),
                label=LabelSpec("fwd_ret_exec_10d", 10), period_start=date(2010, 1, 4), period_end=date(2023, 12, 29))
    return DatasetSpec(**{**base, **over})


SPLIT = SplitSpec("expanding", 500, 60, 60, 10, date(2023, 12, 29))


def _experiment(**over) -> ExperimentSpec:
    base = dict(hypothesis="ridge on f1 factors adds rank IC over order_flow", dataset=_dataset(), split=SPLIT,
                model="ridge", seed=0, baseline="order_flow", primary_test="incremental_rank_ic_newey_west",
                params={"alpha": 1.0})
    return ExperimentSpec(**{**base, **over})


def test_allowlisted_features_are_accepted_at_their_current_version():
    assert F("stock_features", "return_5d").source_version == FEATURE_SET_VERSION
    assert F("stock_factors", "momentum_12_1.z").source_version == f"{factors.FACTOR_SET}:{factors.version()}"
    assert F("market_regimes", "risk").source_version == regimes.version()
    assert len(allowlist.STOCK_FACTORS) == 27 and set(allowlist.MARKET_REGIMES) == set(regimes.DIMENSIONS)


def test_allowlist_is_the_reference_checked_feature_set():
    from tests.research.test_feature_build import FEATURES
    assert set(allowlist.STOCK_FEATURES) == set(FEATURES)


@pytest.mark.parametrize("table, column, why", [
    ("stock_targets", "fwd_ret_exec_5d", "outcomes, never features"),
    ("feature_target", "return_5d", "joins outcomes"),
    ("report_marks", "date", "no publication time"),
    ("fundamental_snapshots", "value", "no publication time"),
    ("interaction_tests", "ic", "Interaction Engine"),
    ("risk_positions", "beta", "portfolio-level"),
    ("cross_features", "leader_excess_1d", "not approved"),
    ("daily_panel", "adj_close", "raw panel"),
    ("some_table", "x", "not an approved point-in-time source"),
    ("stock_features", "session_index", "not in the allowlist"),
    ("stock_features", "is_traded", "not in the allowlist"),
    ("stock_factors", "order_flow.value", "not in the allowlist"),
])
def test_excluded_and_unlisted_sources_are_refused(table, column, why):
    with pytest.raises(ContractError, match=why):
        FeatureSpec(table, column, "v2")


def test_availability_and_source_version_are_checked():
    with pytest.raises(ContractError, match="availability 'close_t\\+1'"):
        FeatureSpec("stock_features", "return_5d", FEATURE_SET_VERSION, "close_t+1")
    with pytest.raises(ContractError, match="definition changed"):
        FeatureSpec("stock_features", "return_5d", "v1")
    with pytest.raises(ContractError, match="definition changed"):
        FeatureSpec("stock_factors", "order_flow.z", "f1:00000000")


def test_label_rules():
    assert LabelSpec("fwd_excess_exec_20d", 20).horizon == 20
    with pytest.raises(ContractError, match="horizon 10, not 5"):
        LabelSpec("fwd_ret_exec_10d", 5)
    with pytest.raises(ContractError, match="not an executable"):
        LabelSpec("fwd_ret_close_5d", 5)                       # close-to-close: not executable
    with pytest.raises(ContractError, match="label convention"):
        LabelSpec("fwd_ret_exec_5d", 5, costs="net")


@pytest.mark.parametrize("end, ok", [(date(2023, 12, 31), True), (date(2024, 1, 1), False)])
def test_research_cutoff_is_an_invariant_of_datasets_and_splits(end, ok):
    assert RESEARCH_END == date(2023, 12, 31)
    if ok:
        _dataset(period_end=end)
        replace(SPLIT, last_date=end)
    else:
        with pytest.raises(ContractError, match="research cutoff"):
            _dataset(period_end=end)
        with pytest.raises(ContractError, match="research cutoff"):
            replace(SPLIT, last_date=end)


def test_dataset_rules():
    with pytest.raises(ContractError, match="duplicate features"):
        _dataset(features=(F("stock_features", "gap"), F("stock_features", "gap")))
    with pytest.raises(ContractError, match="at least one feature"):
        _dataset(features=())
    with pytest.raises(ContractError, match="ordered"):
        _dataset(period_start=date(2015, 1, 5), period_end=date(2014, 1, 6))
    with pytest.raises(ContractError, match="must be a date"):
        _dataset(period_end="2023-12-29")
    with pytest.raises(ContractError, match="must be a date"):
        _dataset(period_end=datetime(2023, 12, 29))
    with pytest.raises(ContractError, match="unknown universe"):
        _dataset(universe="all_listed_today")


def test_split_rules_have_no_shuffle():
    assert "shuffle" not in SplitSpec.__dataclass_fields__
    for kind in ("random", "kfold", "shuffled"):
        with pytest.raises(ContractError, match="is not chronological"):
            replace(SPLIT, kind=kind, train_sessions=750)
    for bad in (dict(kind="random"), dict(kind="kfold"), dict(embargo_sessions=-1), dict(test_sessions=0),
                dict(kind="rolling"), dict(train_sessions=300)):
        with pytest.raises(ContractError):
            replace(SPLIT, **bad)
    replace(SPLIT, kind="rolling", train_sessions=750)


def test_experiment_needs_every_provenance_field():
    e = _experiment()
    assert len(e.experiment_id) == 16
    for bad in (dict(hypothesis=" "), dict(model="xgboost"), dict(seed=-1), dict(baseline="value"),
                dict(primary_test="best_of_grid"), dict(params={"Alpha": 1.0}), dict(params={"alpha": float("nan")})):
        with pytest.raises((ContractError, HashError)):
            _experiment(**bad)


def test_hash_is_canonical_and_versioned():
    assert canonical({"b": 1, "a": [1, 2]}) == canonical({"a": (1, 2), "b": 1})
    assert digest({"d": date(2023, 12, 29)}) == digest({"d": "2023-12-29"})
    assert digest({"x": 1}) != digest({"x": 1}, contract="ml/2") and CONTRACT == "ml/1"
    with pytest.raises(HashError):
        digest({"x": object()})
    with pytest.raises(HashError):
        digest({1: "x"})
    with pytest.raises(HashError):
        digest({"x": float("inf")})


@pytest.mark.parametrize("change", [
    dict(hypothesis="another hypothesis"), dict(model="elasticnet"), dict(seed=1), dict(baseline="liquidity"),
    dict(params={"alpha": 2.0}), dict(split=replace(SPLIT, embargo_sessions=11)),
    dict(split=replace(SPLIT, test_sessions=61)), dict(dataset=_dataset(label=LabelSpec("fwd_ret_exec_5d", 5))),
    dict(dataset=_dataset(period_start=date(2011, 1, 4))),
    dict(dataset=_dataset(features=(F("stock_features", "return_5d"),))),
])
def test_every_meaningful_field_changes_the_experiment_id(change):
    assert _experiment(**change).experiment_id != _experiment().experiment_id


def test_dataset_fingerprint_covers_the_universe_definition(monkeypatch):
    from quant_research.ml import contracts
    before = _dataset().fingerprint
    monkeypatch.setitem(contracts.UNIVERSES, "point_in_time_default", "is_traded AND adv_value_20 > 0")
    assert _dataset().fingerprint != before


def test_same_spec_same_hash_across_processes():
    code = ("from tests.research.test_ml_contracts import _experiment; e = _experiment(); "
            "print(e.experiment_id, e.dataset.fingerprint, e.split.version)")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": seed}).stdout.strip() for seed in ("1", "12345")}
    e = _experiment()
    assert outs == {f"{e.experiment_id} {e.dataset.fingerprint} {e.split.version}"}


def test_duplicate_keys_and_future_feature_rows_are_refused():
    df = pd.DataFrame({"date": [date(2020, 1, 2), date(2020, 1, 2), date(2020, 1, 3)], "symbol": ["A", "B", "A"],
                       "feature_date": [date(2020, 1, 2), date(2020, 1, 2), date(2020, 1, 3)]})
    check_unique(df)
    check_point_in_time(df)
    with pytest.raises(ContractError, match="share a key"):
        check_unique(pd.concat([df, df.iloc[[0]]]))
    late = df.assign(feature_date=[date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 3)])
    with pytest.raises(ContractError, match="after their decision date"):
        check_point_in_time(late)
    with pytest.raises(ContractError, match="undated"):
        check_point_in_time(df.assign(feature_date=[None, date(2020, 1, 2), date(2020, 1, 3)]))

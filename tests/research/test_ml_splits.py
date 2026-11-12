"""Walk-forward folds, interval purge with an embargo gap, and the fold check (docs/ml-plan.md sections 4, 6.1)."""

from dataclasses import replace
from datetime import date, timedelta

import pandas as pd
import pytest

from quant_research.ml.contracts import ContractError, SplitSpec
from quant_research.ml.splits import Fold, check_fold, folds, purge, session_offset


def _weekdays(start: date, n: int, skip: tuple[date, ...] = ()) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5 and d not in skip:
            out.append(d)
        d += timedelta(days=1)
    return out


CAL = _weekdays(date(2023, 1, 2), 30, skip=(date(2023, 1, 23), date(2023, 1, 24)))   # a two-day market holiday
SPLIT = SplitSpec("expanding", 10, 5, 5, 2, date(2023, 1, 2), date(2023, 12, 29))


def test_folds_are_deterministic_chronological_and_complete():
    fs = folds(CAL, SPLIT)
    assert fs == folds(list(reversed(CAL)) + CAL[:3], SPLIT)                # order and duplicates do not matter
    assert [(f.test_start, f.test_end) for f in fs] == [(CAL[10], CAL[14]), (CAL[15], CAL[19]), (CAL[20], CAL[24]),
                                                         (CAL[25], CAL[29])]
    for f in fs:
        assert f.train_start == CAL[0] and f.train_end < f.test_start and f.embargo_start <= f.train_end
    assert fs[0].train_end == CAL[9] and fs[0].embargo_start == CAL[8]      # 2 sessions before the test start
    rolling = folds(CAL, replace(SPLIT, kind="rolling", train_sessions=10))
    assert [f.train_start for f in rolling] == [CAL[0], CAL[5], CAL[10], CAL[15]]


def test_too_few_sessions_or_an_embargo_that_eats_the_training_data_fail():
    with pytest.raises(ContractError, match="too few"):
        folds(CAL[:14], SPLIT)
    with pytest.raises(ContractError, match="leaves no training data"):
        folds(CAL, replace(SPLIT, embargo_sessions=10))


def test_folds_stop_at_the_split_last_date():
    cal = _weekdays(date(2023, 12, 1), 30)                                  # runs into January 2024
    fs = folds(cal, replace(SPLIT, test_sessions=3, step_sessions=3))
    assert max(f.test_end for f in fs) <= date(2023, 12, 29)


def _rows(dates, ends) -> pd.DataFrame:
    return pd.DataFrame({"date": dates, "label_start": [session_offset(CAL, d, 1) for d in dates], "label_end": ends})


def test_purge_uses_the_actual_label_end_not_a_row_count():
    f = folds(CAL, SPLIT)[0]                                                # embargo gap starts at CAL[8]
    # each row decides on CAL[i]; label ends 2 market sessions later, except a suspended symbol whose own
    # sessions skip CAL[4..7]: its label (decided on CAL[3]) ends on CAL[9], inside the gap
    rows = _rows([CAL[3], CAL[3], CAL[5], CAL[6], CAL[9]], [CAL[5], CAL[9], CAL[7], CAL[8], CAL[11]])
    keep = purge(rows, f)
    assert keep.tolist() == [True, False, True, False, False]
    inside = _rows([CAL[12]] * 2, [CAL[14]] * 2)
    assert purge(inside, f).tolist() == [False, False]                          # decided inside the test window


def _fold(test_end: date = date(2023, 12, 29)) -> Fold:
    return Fold(0, date(2023, 1, 2), date(2023, 11, 30), date(2023, 12, 1), test_end, date(2023, 11, 28))


def _frame(dates, starts, ends) -> pd.DataFrame:
    return pd.DataFrame({"date": dates, "label_start": starts, "label_end": ends})


TRAIN = _frame([date(2023, 11, 1)], [date(2023, 11, 2)], [date(2023, 11, 20)])
TEST = _frame([date(2023, 12, 1)], [date(2023, 12, 4)], [date(2023, 12, 20)])


def test_a_valid_fold_passes_and_the_cutoff_day_itself_is_allowed():
    check_fold(TRAIN, TEST, _fold(), SPLIT)
    on_cutoff = _frame([date(2023, 12, 27)], [date(2023, 12, 28)], [date(2023, 12, 31)])
    check_fold(TRAIN, on_cutoff, _fold(date(2023, 12, 31)), replace(SPLIT, last_date=date(2023, 12, 31)))


@pytest.mark.parametrize("train, test, fold, why", [
    (TRAIN, TEST, _fold(date(2024, 1, 1)), "ends after the research cutoff"),
    (TRAIN, _frame([date(2023, 12, 29)], [date(2024, 1, 2)], [date(2024, 1, 3)]), _fold(), "research cutoff"),
    (TRAIN, _frame([date(2024, 1, 2)], [date(2024, 1, 3)], [date(2024, 1, 5)]), _fold(), "research cutoff"),
    (_frame([date(2023, 12, 1)], [date(2023, 12, 4)], [date(2023, 12, 6)]), TEST, _fold(), "outside the training"),
    (TRAIN, _frame([date(2023, 11, 29)], [date(2023, 11, 30)], [date(2023, 12, 5)]), _fold(), "outside the test"),
    (_frame([date(2023, 11, 1)], [date(2023, 11, 2)], [date(2023, 11, 28)]), TEST, _fold(), "embargo gap"),
    (_frame([date(2023, 11, 1)], [date(2023, 11, 1)], [date(2023, 11, 3)]), TEST, _fold(), "does not start after"),
    (_frame([date(2023, 11, 1)], [date(2023, 11, 2)], [None]), TEST, _fold(), "without a label interval"),
    (TRAIN, TEST.iloc[0:0], _fold(), "no test rows"),
])
def test_check_fold_refuses_every_leak(train, test, fold, why):
    with pytest.raises(ContractError, match=why):
        check_fold(train, test, fold, SPLIT)


def test_check_fold_refuses_too_little_training_data():
    with pytest.raises(ContractError, match="training rows < 2"):
        check_fold(TRAIN, TEST, _fold(), replace(SPLIT, min_train_rows=2))


def test_session_offset_counts_market_sessions():
    assert session_offset(CAL, CAL[14], 1) == CAL[15] and CAL[15] - CAL[14] == timedelta(days=5)   # the holiday
    assert session_offset(CAL, CAL[0], 3) == CAL[3] and session_offset(CAL, CAL[-1], 1) is None


def test_purge_boundaries_are_strict():
    f = folds(CAL, SPLIT)[1]                     # test CAL[15]..CAL[19]; embargo gap starts at CAL[13]
    assert (f.embargo_start, f.test_start) == (CAL[13], CAL[15])
    rows = _rows([CAL[9]] * 4, [CAL[12], CAL[13], CAL[14], CAL[16]])
    # ends the session before the gap: kept; on embargo_start, inside the gap, inside the test window: purged
    assert purge(rows, f).tolist() == [True, False, False, False]
    with pytest.raises(ContractError, match="does not start after"):
        purge(rows.assign(label_start=CAL[9]), f)                               # starts on its decision date
    with pytest.raises(ContractError, match="does not start after"):
        purge(rows.assign(label_end=CAL[9]), f)                                 # ends before it starts
    with pytest.raises(ContractError, match="without a label interval"):
        purge(rows.assign(label_end=None), f)


def test_folds_never_use_sessions_before_the_dataset_start():
    split = replace(SPLIT, first_date=CAL[5])
    fs = folds(CAL, split)
    assert fs[0].train_start == CAL[5] and fs[0].test_start == CAL[15]
    assert min(f.train_start for f in folds([date(2022, 12, 1), *CAL], split)) == CAL[5]


def test_purge_drops_rows_decided_before_a_rolling_training_window():
    f = folds(CAL, replace(SPLIT, kind="rolling", train_sessions=10))[1]       # training CAL[5]..CAL[14]
    rows = _rows([CAL[2], CAL[6]], [CAL[4], CAL[8]])
    assert f.train_start == CAL[5] and purge(rows, f).tolist() == [False, True]

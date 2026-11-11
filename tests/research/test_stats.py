"""Statistics module checked against statsmodels and hand-computed values."""

import numpy as np
import pytest
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from quant_research.stats import benjamini_hochberg, benjamini_yekutieli, date_means, mean_test, newey_west_se, summarize


@pytest.mark.parametrize("lag", [0, 1, 4, 9, 19])
def test_newey_west_matches_statsmodels(lag):
    rng = np.random.default_rng(lag)
    x = np.cumsum(rng.normal(size=400)) * 0.01 + rng.normal(size=400)   # autocorrelated
    fit = sm.OLS(x, np.ones_like(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False})
    assert newey_west_se(x, lag) == pytest.approx(float(fit.bse[0]), rel=1e-10)


def test_benjamini_hochberg_matches_statsmodels():
    p = list(np.random.default_rng(1).uniform(size=50) ** 3)
    expected = multipletests(p, method="fdr_bh")[1]
    assert benjamini_hochberg(p) == pytest.approx(list(expected), rel=1e-12)
    assert benjamini_hochberg([]) == []


def test_benjamini_yekutieli_matches_statsmodels_and_a_hand_example():
    p = list(np.random.default_rng(2).uniform(size=50) ** 3)
    expected = multipletests(p, method="fdr_by")[1]
    assert benjamini_yekutieli(p) == pytest.approx(list(expected), rel=1e-12)
    assert all(by >= bh for by, bh in zip(benjamini_yekutieli(p), benjamini_hochberg(p)))
    # m = 3, c(m) = 1 + 1/2 + 1/3 = 11/6; BH q = 0.03 for all three, so BY = 0.03 * 11/6 = 0.055
    assert benjamini_yekutieli([0.01, 0.02, 0.03]) == pytest.approx([0.055] * 3)
    assert benjamini_yekutieli([0.5, 0.9]) == [1.0, 1.0]
    assert benjamini_yekutieli([]) == []


def test_date_means_weight_each_date_equally():
    dates = np.array([2, 1, 1, 1, 2])
    values = np.array([10.0, 1.0, 2.0, 3.0, 20.0])
    uniq, means, counts = date_means(dates, values)
    assert list(uniq) == [1, 2] and list(means) == [2.0, 15.0] and list(counts) == [3, 2]


def test_summary_uses_date_level_mean_and_event_level_descriptives():
    dates = np.array([1, 1, 1, 2, 3])
    values = np.array([0.03, 0.03, 0.03, -0.01, 0.01])
    s = summarize(dates, values, horizon=1, cost=0.004)
    assert (s.n_events, s.n_dates) == (5, 3)
    assert s.mean == pytest.approx(0.01)            # (0.03 - 0.01 + 0.01) / 3
    assert s.event_mean == pytest.approx(0.018)
    assert s.win_rate == pytest.approx(0.8)
    assert s.mean_after_cost == pytest.approx(0.006)


def test_mean_test_is_calibrated_under_the_null_with_overlap():
    """Overlapping 10-day sums of iid noise: about 5% of null samples should be rejected at 5%."""
    rng = np.random.default_rng(42)
    rejections = 0
    for _ in range(400):
        daily = rng.normal(0, 0.02, 760)
        overlapping = np.convolve(daily, np.ones(10), mode="valid")   # strongly autocorrelated series
        _, _, p, _, _ = mean_test(overlapping, lag=9)
        rejections += p < 0.05
    assert 0.02 <= rejections / 400 <= 0.10


def test_mean_test_detects_a_real_effect():
    rng = np.random.default_rng(7)
    _, _, p_effect, lo, _ = mean_test(rng.normal(0.01, 0.02, 500), lag=4)
    assert p_effect < 1e-6 and lo > 0


def test_small_or_empty_samples_return_none():
    assert mean_test(np.array([0.1, 0.2]), lag=1) == (None, None, None, None, None)
    s = summarize(np.array([]), np.array([]), horizon=5, cost=0.004)
    assert s.n_events == 0 and s.mean is None

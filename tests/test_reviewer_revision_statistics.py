import numpy as np
from scipy import stats

from experiments.reviewer_revision_statistics import tost, tost_power, holm


def test_tost_matches_confidence_interval_inclusion():
    x = np.array([-2, -1, 0, 1, 2.])
    for margin in [1., 5.]:
        r = tost(x, margin)
        interval = np.mean(x)+np.array([-1, 1])*stats.t.ppf(.95, 4)*stats.sem(x)
        np.testing.assert_allclose(interval, r['ci_1_minus_2alpha'])
        assert r['equivalence_test_rejects_non_equivalence']==bool(interval[0]>-margin and interval[1]<margin)


def test_exact_power_matches_simulated_paired_tost():
    rng = np.random.default_rng(2); n=12; draws=100000
    x = rng.normal(size=(draws, n)); se=x.std(1, ddof=1)/np.sqrt(n)
    rejects = (np.abs(x.mean(1))+stats.t.ppf(.95, n-1)*se < .8).mean()
    assert abs(rejects-tost_power(n, 1., .8))<.006
    assert tost_power(n, 1., .8)>tost_power(n, 1., .8, true_mean=.5)


def test_holm_monotonicity_and_family_size():
    np.testing.assert_allclose(holm([.03, .001, .5]), [.06, .003, .5])

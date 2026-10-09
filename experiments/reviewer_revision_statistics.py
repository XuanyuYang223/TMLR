"""Paired source-level statistics and power; never count inputs as sources."""
import numpy as np
from scipy import integrate, stats


def tost(differences, margin, alpha=.05):
    differences = np.asarray(differences, float)
    n = len(differences)
    mean = differences.mean(); se = differences.std(ddof=1)/np.sqrt(n)
    assert n>1 and se>0 and margin>0
    lower = stats.t.sf((mean+margin)/se, n-1)
    upper = stats.t.cdf((mean-margin)/se, n-1)
    ci = mean+np.array([-1, 1])*stats.t.ppf(1-alpha, n-1)*se
    return {'mean': float(mean), 'n_sources': n, 'standard_error': float(se),
            'lower_p': float(lower), 'upper_p': float(upper), 'p': float(max(lower, upper)),
            'margin': margin, 'alpha': alpha, 'ci_1_minus_2alpha': ci.tolist(),
            'equivalence_test_rejects_non_equivalence': bool(max(lower, upper)<alpha)}


def tost_power(n, sigma, margin, true_mean=0., alpha=.05):
    """Exact normal-model power by integration over independent sample variance."""
    assert n>1 and sigma>0 and margin>0
    df = n-1; critical = stats.t.ppf(1-alpha, df)
    mean_se = sigma/np.sqrt(n)
    # No TOST rejection is possible beyond this chi-square value.
    limit = df*(margin/(critical*mean_se))**2
    def integrand(v):
        half = margin-critical*mean_se*np.sqrt(v/df)
        probability = stats.norm.cdf((half-true_mean)/mean_se)-stats.norm.cdf((-half-true_mean)/mean_se)
        return max(probability, 0)*stats.chi2.pdf(v, df)
    return float(integrate.quad(integrand, 0, limit, epsabs=1e-9, limit=200)[0])


def required_n(sigma, margin, target=.8, alpha=.05, true_mean=0.):
    for n in range(3, 501):
        if tost_power(n, sigma, margin, true_mean, alpha)>=target:
            return n
    return None


def holm(pvalues):
    values = np.asarray(pvalues, float); order = np.argsort(values)
    adjusted = np.minimum(1, np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
    output = np.empty_like(values); output[order] = adjusted
    return output.tolist()

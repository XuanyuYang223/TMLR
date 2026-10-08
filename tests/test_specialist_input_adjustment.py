import numpy as np
from experiments.specialist_input_adjustment import adjust, alignment


def test_psd_adjustment_removes_negative_eigenvalues():
    k=np.diag([5.,1.,.1]);g=np.diag([1.,1.,1.]);psd,alpha,negative=adjust(k,g)
    np.testing.assert_allclose(alpha,6.1/3)
    assert np.linalg.eigvalsh(psd).min()>=-1e-12 and negative>0


def test_adjusted_alignment_respects_feature_scale():
    rng=np.random.default_rng(3);x=rng.normal(size=(12,4));y=rng.normal(size=(12,4));z=rng.normal(size=(12,4))
    g=z@z.T;k=x@x.T;l=y@y.T
    a,_,_=adjust(k,g);b,_,_=adjust(l,g);scaled,_,_=adjust(9*k,g)
    np.testing.assert_allclose(alignment(a,b),alignment(scaled,b),atol=1e-12)

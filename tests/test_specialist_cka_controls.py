import numpy as np
from experiments.specialist_cka_controls import center_groups, cka, common_strata, confidence_residual, original_input_key, input_key


def test_original_compact_and_spaced_inputs():
    for line in [b'{"inputs":{"primary":[2,3,1]}}',b'{"inputs": {"primary": [2, 3, 1]}}']:
        assert original_input_key(line)==input_key([2,3,1])


def test_constant_stratum_is_undefined():
    x=np.eye(3)[[0,0,0,1,1,1]];keys=np.array([[0]]*3+[[1]]*3)
    residual=center_groups(x,keys)
    assert cka(residual,residual) is None


def test_shared_strata_preserve_all_answers():
    n=np.array([10]*7);left=np.array([1]*3+[2]*3+[3]);right=np.column_stack([left,left+1,left+2])
    keys,keep,audit=common_strata(n,left,right,3)
    assert audit['retained']==6 and keep.tolist()==[True]*6+[False]
    np.testing.assert_array_equal(center_groups(right[keep],keys[keep]),0)


def test_cka_orthogonal_invariance():
    x=np.random.default_rng(1).normal(size=(50,6));q,_=np.linalg.qr(np.random.default_rng(2).normal(size=(6,6)))
    assert abs(cka(x,x@q)-1)<1e-12


def test_confidence_fit_does_not_use_test_features():
    rng=np.random.default_rng(1);x=rng.normal(size=(20,4));p=np.ones((20,31))/31
    c=rng.normal(size=(20,3));y=np.arange(20)%5;n=np.full(20,10)
    fit=np.arange(20)<10;test=~fit
    a=confidence_residual(x,p,c,y,n,fit,test,.01)
    changed=x.copy();changed[test]+=7
    b=confidence_residual(changed,p,c,y,n,fit,test,.01)
    np.testing.assert_allclose(b-a,7)

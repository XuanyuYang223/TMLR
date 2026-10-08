import numpy as np

from experiments.algebra_relation_worlds import (
    action, answer, create_world, generators, group, key, matrix_product, transform,
)


def test_matrix_group_laws_and_noncommuting_order():
    p=13;a,b=generators(p);identity=(1,0,0,1)
    assert len(group(p))==6
    assert matrix_product(a,a,p)==identity
    assert matrix_product(matrix_product(b,b,p),b,p)==identity
    assert matrix_product(matrix_product(a,b,p),a,p)==matrix_product(b,b,p)
    assert matrix_product(a,b,p)!=matrix_product(b,a,p)


def test_matrix_collision_has_unknown_compound_answer():
    p=13;left=(1,0,0,1);right=(12,1,12,3)
    assert all((x[0]*x[3]-x[1]*x[2])%p for x in [left,right])
    assert [answer(transform(left,w,'matrix',p),'matrix',p) for w in ['', 'a','b']]==[
        answer(transform(right,w,'matrix',p),'matrix',p) for w in ['', 'a','b']]
    assert answer(transform(left,'ab','matrix',p),'matrix',p)==0
    assert answer(transform(right,'ab','matrix',p),'matrix',p)==3
    assert key(left,'matrix',p)!=key(right,'matrix',p)


def test_polynomial_shift_agrees_with_direct_evaluation():
    p=13;rng=np.random.default_rng(811)
    for _ in range(10):
        c=tuple(map(int,rng.integers(p,size=4)));shifted=action(c,'a','polynomial',p)
        derivative=action(c,'b','polynomial',p)
        for x in range(p):
            expected=sum(c[j]*pow(x+1,j,p) for j in range(4))%p
            assert sum(shifted[j]*pow(x,j,p) for j in range(4))%p==expected
            assert sum(derivative[j]*pow(x,j,p) for j in range(4))%p==sum(j*c[j]*pow(x,j-1,p) for j in range(1,4))%p


def test_polynomial_commutation_nilpotency_and_user_collision():
    p=13;left=(0,1,0,0);right=(0,1,12,1)
    assert [answer(transform(left,w,'polynomial',p),'polynomial',p) for w in ['', 'a','b']]==[0,1,1]
    assert [answer(transform(right,w,'polynomial',p),'polynomial',p) for w in ['', 'a','b']]==[0,1,1]
    assert transform(left,'ab','polynomial',p)[0]==1
    assert transform(right,'ab','polynomial',p)[0]==2
    for c in [left,right,(5,3,2,7)]:
        assert transform(c,'ab','polynomial',p)==transform(c,'ba','polynomial',p)
        assert transform(c,'bbbb','polynomial',p)==(0,0,0,0)
        assert transform(c,'a'*p,'polynomial',p)==c


def test_polynomial_split_block_preserves_translation_and_derivative_orbit():
    p=13;c=(4,5,8,3);k=key(c,'polynomial',p)
    for i in range(p):
        moved=transform(c,'a'*i,'polynomial',p)
        assert key(moved,'polynomial',p)==k
        derivative=action(moved,'b','polynomial',p)
        assert derivative[2]==3*k[0]%p
        assert (derivative[1]**2-4*derivative[0]*derivative[2])%p==k[1]


def test_actual_paths_are_disjoint_in_both_worlds():
    for domain in ['matrix','polynomial']:
        data=create_world(domain,fit_count=32,source_count=8,validation_count=4,iid_count=8,pair_count=4)
        train=set(tuple(map(int,x)) for d in [data['source']]+data['supports'] for x in d['x'].reshape(-1,4))
        test=set(tuple(map(int,x)) for x in data['test']['x'].reshape(-1,4))
        assert not train&test
        for pair in range(4):
            ids=np.flatnonzero(data['test']['pair_ids']==pair)
            np.testing.assert_array_equal(data['test']['labels'][ids[0],:3],data['test']['labels'][ids[1],:3])
            assert data['test']['labels'][ids[0],3]!=data['test']['labels'][ids[1],3]

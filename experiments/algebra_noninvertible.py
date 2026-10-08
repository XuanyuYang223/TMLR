"""Exact non-group operators, using COLUMN coefficient vectors.

Coefficients are ordered constant term first. All operators preserve the
ambient degree-(d+1) coefficient space; truncation zeroes high coefficients.
This differs from dimension-changing deletion of coordinates.
"""
import numpy as np


def coordinate_zero_projection(dimension,index):
    if not 0<=index<dimension:raise ValueError('index outside ambient space')
    p=np.eye(dimension,dtype=np.int64);p[index,index]=0
    return p


def polynomial_operators(max_degree):
    if max_degree<0:raise ValueError('degree must be nonnegative')
    dimension=max_degree+1;derivative=np.zeros((dimension,dimension),dtype=np.int64)
    for j in range(1,dimension):derivative[j-1,j]=j
    projections=[np.diag(np.arange(dimension)<=k).astype(np.int64) for k in range(dimension)]
    return derivative,projections


def polynomial_law_audit(max_degree=8):
    tested=0
    for degree in range(max_degree+1):
        derivative,projections=polynomial_operators(degree)
        assert not np.linalg.matrix_power(derivative,degree+1).any();tested+=1
        for j,pj in enumerate(projections):
            assert np.array_equal(pj @ pj,pj);tested+=1
            for k,pk in enumerate(projections):
                assert np.array_equal(pj @ pk,projections[min(j,k)]);tested+=1
            if j>=1:
                assert np.array_equal(derivative @ pj,projections[j-1] @ derivative);tested+=1
    return {'operator_convention':'column coefficient vectors; fixed ambient dimension',
        'max_degree':max_degree,'exact_matrix_laws_checked':tested,
        'relations':['P_j P_k = P_min(j,k)','D P_k = P_(k-1) D for k>=1','D^(d+1) = 0 for degree<=d'],
        'neural_models_trained':False}

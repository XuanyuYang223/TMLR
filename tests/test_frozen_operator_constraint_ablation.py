import numpy as np
import torch
from experiments.frozen_operator_constraint_ablation import apply_twice, project


def test_affine_roundtrip_includes_bias():
    x=torch.tensor([[1.,2.],[3.,4.]])
    a=torch.diag(torch.tensor([-1.,1.]));b=torch.tensor([6.,0.])
    torch.testing.assert_close(apply_twice(x,a,b),x)
    assert not torch.allclose(apply_twice(x,a,torch.tensor([0.,2.])),x)


def test_spectral_projection_preserves_eligible_singular_values():
    a=torch.tensor([[[4.,0.],[0.,.5]],[[2.,0.],[0.,1.]]])
    project(a,1.)
    torch.testing.assert_close(a,torch.tensor([[[1.,0.],[0.,.5]],[[1.,0.],[0.,1.]]]))


def test_centered_affine_map_returns_to_original_coordinates():
    rng=np.random.default_rng(1);x=rng.normal(size=(5,4));mu=rng.normal(size=4);a=rng.normal(size=(2,4,4));b=rng.normal(size=(2,4));s=3.
    normalized_bias=(mu@a+b-mu)/s
    recovered_bias=s*normalized_bias+mu-mu@a
    np.testing.assert_allclose(recovered_bias,b)
    for i in range(2):np.testing.assert_allclose(((x-mu)/s@a[i]+normalized_bias[i])*s+mu,x@a[i]+b[i])

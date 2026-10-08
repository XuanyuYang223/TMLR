import numpy as np
import torch

from experiments.final_mechanism_solver import solve,objective_and_gradient,affine_from_identity
from experiments.operator_capacity_confirmation import MatchedResidual


def solver_config():
    return {'solver':{'svd_relative_rank_threshold':1e-12,'max_iterations':100,
            'gradient_tolerance':1e-9,'change_tolerance':1e-14,'gap_bound_tolerance':1e-8},
            'temperature':2.,'geometry_weight':.25,'output_kd_weight':1.}


def test_analytic_shared_loss_gradient_matches_finite_difference():
    rng=np.random.default_rng(54);p=rng.normal(size=(9,4));target=rng.normal(size=p.shape)
    w=rng.normal(size=(3,4));b=rng.normal(size=3);teacher=rng.normal(size=(9,3))
    cfg=solver_config();value,g=objective_and_gradient(p,target,teacher,w,b,1.7,cfg)
    direction=rng.normal(size=p.shape);eps=1e-6
    plus=objective_and_gradient(p+eps*direction,target,teacher,w,b,1.7,cfg)[0]
    minus=objective_and_gradient(p-eps*direction,target,teacher,w,b,1.7,cfg)[0]
    np.testing.assert_allclose((plus-minus)/(2*eps),(g*direction).sum(),atol=1e-8,rtol=1e-8)


def test_same_affine_objective_converges_from_both_fixed_starts():
    torch.set_num_threads(1);rng=np.random.default_rng(25)
    h=rng.normal(size=(40,4));x=np.column_stack([h,np.ones(40)])
    theta=rng.normal(size=(5,4));target=x@theta
    w=rng.normal(size=(3,4));b=rng.normal(size=3);teacher=target@w.T+b
    a,meta=solve(h,target,teacher,w,b,1.7,solver_config())
    c,second=solve(h,target,teacher,w,b,1.7,solver_config(),np.zeros_like(theta))
    assert meta['converged'] and second['converged']
    np.testing.assert_allclose(x@a,target,atol=1e-8)
    np.testing.assert_allclose(x@c,target,atol=3e-5)


def test_two_layer_identity_and_direct_affine_are_same_function_family():
    torch.manual_seed(44);m=MatchedResidual(np.eye(4),np.zeros(4),8,False).double()
    with torch.no_grad():m.layers[2].weight.normal_();m.layers[2].bias.normal_()
    h=torch.randn(12,4,dtype=torch.float64)
    theta=affine_from_identity(m.state_dict())
    x=np.column_stack([h.numpy(),np.ones(len(h))])
    np.testing.assert_allclose(m(h).detach().numpy(),x@theta,atol=1e-12)

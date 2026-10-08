import numpy as np
import torch

from experiments.inverse_alignment_coverage import losses,schedules
from experiments.inverse_functional_alignment import matched_derangement

PLAN = {'geometry_weight':.3,'distillation_weight':1.,'distillation_temperature':2.,'updates':20,'lengths':[10,20],'labeled_batch':4,'unlabeled_batch':4}


def fixture():
    torch.manual_seed(11)
    return torch.randn(8,7,requires_grad=True),torch.randn(8,5,requires_grad=True),torch.tensor([1,2,3,1]),torch.randn(8,5,requires_grad=True),torch.randn(8,5,requires_grad=True),torch.randn(8,7,requires_grad=True),torch.ones(8,dtype=torch.bool)


def test_exposure_only_unlabeled_rows_receive_no_learning_gradient():
    args=fixture();loss,_=losses(*args,4,'ordinary_exposure',PLAN);loss.backward()
    assert torch.count_nonzero(args[0].grad[:4]) and not torch.count_nonzero(args[0].grad[4:])
    assert not torch.count_nonzero(args[1].grad) and all(x.grad is None for x in args[3:6])


def test_support_and_coverage_geometry_have_different_gradient_scope():
    args=fixture();loss,_=losses(*args,4,'support_alignment',PLAN);loss.backward()
    assert torch.count_nonzero(args[1].grad[:4]) and not torch.count_nonzero(args[1].grad[4:])
    args=fixture();loss,_=losses(*args,4,'coverage_alignment',PLAN);loss.backward()
    assert torch.count_nonzero(args[1].grad[4:]) and args[3].grad is None


def test_pool_matching_requires_only_teacher_prediction_and_length():
    n=np.array([10]*4+[20]*4);pseudo=np.array([2,2,3,3,4,4,4,5]);p,e=matched_derangement(n,pseudo,np.ones(8,dtype=bool),12)
    assert e.sum()==7 and not e[-1];np.testing.assert_array_equal(n[e],n[p[e]]);np.testing.assert_array_equal(pseudo[e],pseudo[p[e]])


def test_schedule_has_equal_lengths_and_no_validation_rows():
    support={'lengths':np.array([10]*8+[20]*8),'split':np.array([0]*6+[1]*2+[0]*6+[1]*2)};pool={'lengths':np.array([10]*20+[20]*20)}
    a,b=schedules(PLAN,{'data_seed':23},support,pool,31)
    assert np.all(support['split'][a]==0) and np.all(support['lengths'][a]==pool['lengths'][b])
    aa,bb=schedules(PLAN,{'data_seed':23},support,pool,31);np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(b,bb)

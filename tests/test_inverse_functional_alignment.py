import numpy as np
import torch
from experiments.inverse_functional_alignment import matched_derangement, geometry_loss, output_loss


def test_mismatch_preserves_length_answer_and_training_scope():
    n=np.array([10,10,10,10,20,20,10,20]);y=np.array([2,2,3,3,2,2,2,2]);mask=np.array([True]*6+[False]*2)
    partner,eligible=matched_derangement(n,y,mask,3)
    assert eligible.sum()==6 and np.all(partner[eligible]!=np.flatnonzero(eligible))
    np.testing.assert_array_equal(n[eligible],n[partner[eligible]])
    np.testing.assert_array_equal(y[eligible],y[partner[eligible]])
    assert not np.any(eligible[~mask]) and not np.any(np.isin(partner[eligible],np.flatnonzero(~mask)))


def test_singleton_is_excluded_from_both_geometry_conditions():
    partner,eligible=matched_derangement(np.array([10,10,10]),np.array([1,1,2]),np.ones(3,dtype=bool),1)
    assert eligible.tolist()==[True,True,False] and partner.tolist()==[1,0,2]


def test_geometry_loss_is_orthogonally_invariant_and_teacher_frozen():
    torch.manual_seed(4);x=torch.randn(12,6,requires_grad=True);q,_=torch.linalg.qr(torch.randn(6,6));y=(x.detach()@q).requires_grad_()
    loss=geometry_loss(x,y);assert abs(float(loss.detach()))<1e-6
    loss.backward();assert x.grad is not None and y.grad is None


def test_geometry_detects_wrong_pairing_with_same_answers():
    torch.manual_seed(8);x=torch.randn(20,5);y=x.clone()
    p=torch.tensor([*range(1,10),0,*range(11,20),10])
    assert float(geometry_loss(x,y[p]))>float(geometry_loss(x,y))+.1


def test_distillation_has_no_teacher_gradient():
    torch.manual_seed(9);student=torch.randn(6,8,requires_grad=True);teacher=torch.randn(6,8,requires_grad=True)
    loss=output_loss(student,teacher,2.);loss.backward()
    assert student.grad is not None and teacher.grad is None

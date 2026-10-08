import torch
from experiments.inverse_coverage_size_control import equal_count_loss


def test_equal_count_alignment_uses_same_count_but_only_unlabeled_gradient():
    torch.manual_seed(1);h=torch.randn(8,5,requires_grad=True);teacher=torch.randn(8,5,requires_grad=True)
    loss,count=equal_count_loss(h,teacher,torch.tensor([True,False,True,True]),4);loss.backward()
    assert count==3 and not torch.count_nonzero(h.grad[:4]) and not torch.count_nonzero(h.grad[7:])
    assert torch.count_nonzero(h.grad[4:7]) and teacher.grad is None

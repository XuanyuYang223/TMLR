import numpy as np
from experiments.downstream_harm_diagnostic_v2 import metrics
from experiments.relation_error_localization import row_projection


def test_output_invisible_error_can_flip_compound_classification():
    # First readout cannot see coordinate 2; second operator moves it to coord 1.
    w=np.array([[1.,0.],[-1.,0.]])
    b=np.array([[1.,0.],[-2.,1.]])
    h=np.array([[1.,0.]])
    e=np.array([[0.,1.]])
    q=np.eye(2)-row_projection(w)
    np.testing.assert_allclose((e@q)@w.T,0,atol=1e-12)
    z=h@b@w.T;d=e@q@b@w.T
    result=metrics(z,d,np.array([0]))
    assert result['forced_correct'][0]
    assert not result['perturbed_correct'][0]
    assert result['worst_margin_ratio'][0]==2.


def test_margin_statistic_ignores_self_class_and_common_logit_shift():
    z=np.array([[3.,1.,0.],[0.,2.,1.]])
    y=np.array([0,1]);d=np.array([[0.,1.,2.],[1.,0.,0.]])
    result=metrics(z,d,y)
    shifted=metrics(z,d+23.,y)
    assert np.isfinite(result['worst_margin_ratio']).all()
    np.testing.assert_allclose(result['worst_margin_ratio'],[2/3,1/2])
    for key in ['worst_margin_ratio','centered_logit_sq','perturbed_correct']:
        np.testing.assert_allclose(result[key],shifted[key])


def test_null_loss_ignores_readout_rows_but_preserves_output_losses(monkeypatch):
    import torch
    from torch import nn
    from experiments import readout_null_permworld_v2 as study
    from experiments.hidden_relation_train import AffineOperators
    model=nn.Module();model.lm_head=nn.Linear(4,31,bias=False)
    with torch.no_grad():
        model.lm_head.weight.zero_();model.lm_head.weight[:2,:2]=torch.eye(2)
    model.lm_head.weight.requires_grad_(False)
    monkeypatch.setattr(study,'query_hidden',lambda model,inputs,lengths,task,tokens:inputs[:,:4].float())
    x=torch.zeros(3,3,4);x[:,1,0]=1.;x[:,2,1]=2.
    y=torch.zeros(3,3,dtype=torch.long);teacher=torch.zeros(3,3,31)
    ids=torch.arange(3);lengths=torch.full((3,),12);ops=AffineOperators(4)
    ops.geometry_projector=torch.diag(torch.tensor([0.,0.,1.,1.]));ops.geometry_scale=1.
    ops.geometry_space='full'
    full=study.joint_losses(model,ops,x,lengths,y,teacher,ids,ids,ids,'task',{},2.,1.)
    ops.geometry_space='null'
    null=study.joint_losses(model,ops,x,lengths,y,teacher,ids,ids,ids,'task',{},2.,1.)
    for a,b in zip(full[:3],null[:3]):torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert full[3]>0
    torch.testing.assert_close(null[3],torch.tensor(0.))

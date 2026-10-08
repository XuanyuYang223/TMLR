"""Same-loss, same-affine-family convergence checks with fixed initializations."""
import time
import numpy as np
import torch
from torch.nn import functional as F


def affine_from_identity(state):
    w1,b1=state['layers.0.weight'].cpu().numpy().astype(float),state['layers.0.bias'].cpu().numpy().astype(float)
    w2,b2=state['layers.2.weight'].cpu().numpy().astype(float),state['layers.2.bias'].cpu().numpy().astype(float)
    a=state['a'].cpu().numpy().astype(float)+w1.T@w2.T
    b=state['bias'].cpu().numpy().astype(float)+b1@w2.T+b2
    return np.vstack([a,b])


def objective_and_gradient(pred, target, teacher_logits, w, bw, scale, config):
    def logsoft(z):
        z=z-z.max(-1,keepdims=True)
        return z-np.log(np.exp(z).sum(-1,keepdims=True))
    t=config['temperature'];lt=logsoft(teacher_logits/t);lp=logsoft((pred@w.T+bw)/t)
    pt,pp=np.exp(lt),np.exp(lp)
    kd=float((pt*(lt-lp)).sum(-1).mean()*t*t)
    geom=float(np.square(pred-target).mean()/scale)
    derivative=config['output_kd_weight']*t*(pp-pt)@w/len(pred)
    derivative+=2*config['geometry_weight']*(pred-target)/(pred.size*scale)
    return config['output_kd_weight']*kd+config['geometry_weight']*geom,derivative


def solve(h,target,teacher_logits,w,bw,scale,config,initial_theta=None):
    x=np.column_stack([h,np.ones(len(h))])
    u,s,v=np.linalg.svd(x,full_matrices=False)
    assert np.all(s>s[0]*config['solver']['svd_relative_rank_threshold'])
    transform=v.T*(np.sqrt(len(x))/s)[None,:];features=u*np.sqrt(len(x))
    _,sw,vw=np.linalg.svd(w,full_matrices=False);rb=vw[sw>sw[0]*1e-10]
    c_ls=features.T@target/len(x);null=c_ls-c_ls@rb.T@rb
    start=c_ls if initial_theta is None else features.T@(x@initial_theta)/len(x)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    ts=lambda z:torch.as_tensor(z,dtype=torch.float64,device=device)
    fx,ty,nu,rows,wt,bwt,teacher=[ts(z) for z in [features,target,null,rb,w,bw,teacher_logits]]
    temperature=config['temperature'];tl=F.log_softmax(teacher/temperature,-1);tp=tl.exp()
    param=torch.nn.Parameter(ts(start@rb.T));calls=0
    opt=torch.optim.LBFGS([param],max_iter=config['solver']['max_iterations'],history_size=100,
        tolerance_grad=config['solver']['gradient_tolerance'],tolerance_change=config['solver']['change_tolerance'],line_search_fn='strong_wolfe')
    def closure():
        nonlocal calls
        calls+=1;opt.zero_grad(set_to_none=True)
        pred=fx@(nu+param@rows)
        logprob=F.log_softmax((pred@wt.T+bwt)/temperature,-1)
        kd=(tp*(tl-logprob)).sum(-1).mean()*temperature**2
        geometry=(pred-ty).square().mean()/scale
        loss=config['output_kd_weight']*kd+config['geometry_weight']*geometry
        loss.backward();return loss
    began=time.monotonic();opt.step(closure);loss=closure()
    coef=null+param.detach().cpu().numpy()@rb
    theta=transform@coef
    grad=param.grad.cpu().numpy();mu=2*config['geometry_weight']/(scale*target.shape[1])
    gap=float(np.square(grad).sum()/(2*mu))
    actual=x@theta
    independent_loss,derivative=objective_and_gradient(actual,target,teacher_logits,w,bw,scale,config)
    full_grad=features.T@derivative
    independent_gap=float(np.square(full_grad).sum()/(2*mu))
    return theta,{'objective':float(loss.detach()),'independent_objective':independent_loss,
        'gradient_norm':float(np.linalg.norm(grad)),'gap_bound':gap,'independent_gap_bound':independent_gap,
        'converged':max(gap,independent_gap)<=config['solver']['gap_bound_tolerance'],
        'iterations':int(opt.state[param]['n_iter']),'closure_calls':calls,'elapsed_seconds':time.monotonic()-began,
        'condition_number':float(s[0]/s[-1]),'singular_values':s.tolist(),'coefficient_norm':float(np.linalg.norm(theta)),
        'same_unregularized_loss':True,'same_affine_function_family':True,'no_composite_training':True,
        'initialization':'ols' if initial_theta is None else 'budget_linear',
        'state_only_linear_minimum_nmse':float(np.square(features@c_ls-target).mean()/scale)}

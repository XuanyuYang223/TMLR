"""Fixed complementary geometry metrics; no source fitting or optimizer."""
import numpy as np


def center(x,keys=None):
    x=np.asarray(x,np.float64).copy()
    if not len(x):return x
    if keys is None:return x-x.mean(0)
    _,inv=np.unique(keys,axis=0,return_inverse=True)
    for g in np.unique(inv):
        use=inv==g;x[use]-=x[use].mean(0)
    return x


def align(k,l):
    den=np.linalg.norm(k)*np.linalg.norm(l)
    return float(np.sum(k*l)/den) if den>1e-20 else None


def cka(x,y,debiased=False):
    if len(x)<4 and debiased:return None
    if len(x)<2:return None
    x=center(x);y=center(y);k=x@x.T;l=y@y.T
    if debiased:
        for a in [k,l]:
            n=len(a);np.fill_diagonal(a,0);s=a.sum(1)
            a-=s[:,None]/(n-2)+s[None,:]/(n-2)
            a+=s.sum()/((n-1)*(n-2));np.fill_diagonal(a,0)
    return align(k,l)


def neighbors(x,k=10,ids=None,duplicate_keys=None):
    n=len(x)
    if n<=k or np.square(center(x)).sum()<=1e-20:return None
    ids=np.arange(n) if ids is None else np.asarray(ids)
    # Sort sample IDs first, then stable distance order for exact ties.
    order=np.argsort(ids,kind='stable');x=np.asarray(x,np.float64)[order]
    d=np.maximum(np.square(x).sum(1)[:,None]+np.square(x).sum(1)[None,:]-2*x@x.T,0)
    np.fill_diagonal(d,np.inf)
    if duplicate_keys is not None:
        keys=np.asarray(duplicate_keys)[order]
        same=np.all(keys[:,None,:]==keys[None,:,:],axis=-1)
        d[same]=np.inf
    if np.any(np.isfinite(d).sum(1)<k):return None
    chosen=np.argsort(d,axis=1,kind='stable')[:,:k]
    sets=[set(ids[order][r].tolist()) for r in chosen]
    return [sets[i] for i in np.argsort(order)]


def knn_overlap(x,y,k=10,ids=None,duplicate_keys=None):
    a=neighbors(x,k,ids,duplicate_keys);b=neighbors(y,k,ids,duplicate_keys)
    if a is None or b is None:return None
    return float(np.mean([len(u&v)/k for u,v in zip(a,b)]))


def pca_fit(x,r):
    x=np.asarray(x,np.float64)
    if len(x)<=r or np.square(x).sum()<=1e-20:return None
    _,s,vt=np.linalg.svd(x,full_matrices=False)
    if np.sum(s>s[0]*1e-6)<r:return None
    basis=vt[:r].T;z=x@basis;scale=np.sqrt(np.square(z).sum()/len(z))
    return (basis,scale) if scale>1e-12 else None


def procrustes(xfit,yfit,xtest,ytest,rank_cap=8):
    r=min(rank_cap,xfit.shape[1],yfit.shape[1])
    a=pca_fit(xfit,r);b=pca_fit(yfit,r)
    if a is None or b is None or len(xtest)<2:return None,r
    bx,sx=a;by,sy=b
    xf=xfit@bx/sx;yf=yfit@by/sy
    u,_,vt=np.linalg.svd(xf.T@yf,full_matrices=False);rotation=u@vt
    xx=xtest@bx/sx;yy=ytest@by/sy;den=np.square(yy).sum()
    if den<=1e-20:return None,r
    error=float(np.sqrt(np.square(xx@rotation-yy).sum()/den))
    return error,r


def scores(xfit,yfit,xtest,ytest,ids=None,duplicate_keys=None):
    error,r=procrustes(xfit,yfit,xtest,ytest)
    return {'linear_cka':(cka(xtest,ytest),None),
            'debiased_cka':(cka(xtest,ytest,True),None),
            'knn_overlap':(knn_overlap(xtest,ytest,ids=ids,duplicate_keys=duplicate_keys),None),
            'procrustes':(-error if error is not None else None,error)},r


def verify():
    rng=np.random.default_rng(2026100942);x=rng.normal(size=(40,3));xt=rng.normal(size=(40,3))
    q=np.linalg.qr(rng.normal(size=(3,3)))[0]
    y=2*x@q;yt=2*xt@q
    assert abs(cka(x,y)-1)<1e-12
    assert knn_overlap(x,y)==1
    e,r=procrustes(center(x),center(y),center(xt),center(yt))
    assert r==3 and e<1e-12
    constant=np.ones((40,3));assert cka(constant,constant) is None
    assert knn_overlap(constant,constant) is None
    assert procrustes(center(constant),center(constant),center(constant),center(constant))[0] is None
    a=np.array([0.,1.,3.,10.])[:,None];b=np.array([0.,3.,1.,10.])[:,None]
    assert knn_overlap(a,b,k=1)==0
    assert neighbors(np.array([0.,1.,-1.])[:,None],k=1,ids=[5,2,9])[0]=={2}
    # Independent cross-covariance formula verifies Gram CKA on random input.
    x=center(x);y=center(rng.normal(size=(40,5)))
    independent=np.square(x.T@y).sum()/np.sqrt(np.square(x.T@x).sum()*np.square(y.T@y).sum())
    assert abs(cka(x,y)-independent)<1e-12
    assert cka(np.ones((3,2)),np.ones((3,2)),True) is None
    return {'cka_rotation_and_independent_formula':True,'constant_undefined':True,
        'heldout_rotation_error_zero':True,'manual_neighbor_swap_overlap_zero':True,'exact_tie_by_input_id':True}

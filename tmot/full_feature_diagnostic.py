"""Arrived-data refit diagnostic, NOT a bounded-memory continual learner."""
import numpy as np


def unprojected_features(basis,x):
    """Retain every coordinate with expected branch energy of the 512-D map."""
    x=np.asarray(x,np.float32);d=basis.semantic_dim
    semantic=(x[:,:d]-basis.mean)/basis.scale
    amplitude=(x[:,d:]-basis.amp_mean)/basis.amp_sd
    return np.concatenate([semantic*np.sqrt((basis.dimension-basis.amp_width)/d),
                           amplitude*np.sqrt(basis.amp_width/basis.amplitude_dim)],axis=1).astype(np.float32)


def weighted_ridge_grid(x,y,weights,ridges):
    """Exact weighted ridge, unpenalized bias; share eigendecomposition over grid."""
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64);w=np.asarray(weights,np.float64)
    if y.ndim==1:y=y[:,None]
    if x.ndim!=2 or len(x)!=len(y) or w.shape!=(len(x),):raise ValueError('Shape mismatch')
    if not all(np.isfinite(a).all() for a in (x,y,w)) or np.any(w<=0):raise ValueError('Invalid data/weights')
    if any(r<=0 for r in ridges):raise ValueError('Positive ridge required')
    w=w/w.sum();xm=w@x;ym=w@y;xx=(x-xm)*np.sqrt(w[:,None]);yy=(y-ym)*np.sqrt(w[:,None])
    dual=len(x)<x.shape[1]
    gram=xx@xx.T if dual else xx.T@xx
    eig,q=np.linalg.eigh((gram+gram.T)/2)
    if eig.min() < -1e-9*max(1,float(eig.max())):raise ValueError('Non-PSD Gram')
    eig=np.maximum(eig,0)
    left=xx.T@q if dual else q;right=q.T@yy if dual else q.T@(xx.T@yy)
    result={}
    for ridge in ridges:
        coef=left@(right/(eig[:,None]+ridge));intercept=ym-xm@coef
        result[float(ridge)]=(coef,intercept)
    return result,dict(solver='dual' if dual else 'primal',system_dimension=len(eig),input_dimension=x.shape[1],samples=len(x))

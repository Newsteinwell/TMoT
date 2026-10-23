"""Finite regression replay calibration using only historical sufficient stats.

Matches target mean/covariance, XY covariance and input mean/marginal variance.
Does not preserve full XX covariance or guarantee nonlinear-risk equivalence.
"""
import numpy as np


def calibrate_regression(x,y,memory,rng):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if y.ndim==1:y=y[:,None]
    n=len(x);g,k,_=memory.risk_moments()
    mx=g[:-1,-1];my=k[-1]
    cross=k[:-1]-np.outer(mx,my)
    yy=memory.target_gram/memory.n-np.outer(my,my);yy=(yy+yy.T)/2
    variance=np.maximum(np.diag(g)[:-1]-mx**2,0)
    values,vectors=np.linalg.eigh(yy)
    keep=values>max(float(values.max(initial=0))*1e-12,1e-15)
    values,vectors=values[keep],vectors[:,keep]
    rank=len(values)
    if n<rank+2:raise ValueError('Replay count must exceed target rank plus one')
    factor=vectors*np.sqrt(values)
    precision=(vectors/values)@vectors.T if rank else np.zeros_like(yy)
    yc=y-y.mean(0)
    if rank:
        z=yc@vectors;z-=z.mean(0)
        sample_cov=z.T@z/n;eig,vec=np.linalg.eigh(sample_cov)
        if eig.min()<=max(eig.max()*1e-12,1e-15):
            # Degenerate generated targets: fresh centered orthogonal draws.
            z=rng.normal(size=(n,rank));z-=z.mean(0)
            z=np.linalg.qr(z,mode='reduced')[0]*np.sqrt(n)
        else:z=z@((vec/np.sqrt(eig))@vec.T)
        yc=z@factor.T
    else:yc=np.zeros_like(y)
    xc=x-x.mean(0)
    residual=xc-yc@(precision@(yc.T@xc/n))
    residual-=residual.mean(0)
    beta=precision@cross.T
    required=np.maximum(variance-np.sum(beta*(yy@beta),axis=0),0)
    observed=np.mean(residual**2,axis=0)
    missing=(observed<1e-15)&(required>1e-12)
    if missing.any():
        replacement=rng.normal(size=(n,int(missing.sum())));replacement-=replacement.mean(0)
        replacement-=yc@(precision@(yc.T@replacement/n))
        residual[:,missing]=replacement
        observed=np.mean(residual**2,axis=0)
    scale=np.sqrt(required/np.maximum(observed,1e-30))
    calibrated_x=(mx+yc@beta+residual*scale).astype(np.float32)
    calibrated_y=(my+yc).astype(np.float32)
    cx=calibrated_x.astype(float)-calibrated_x.astype(float).mean(0)
    cy=calibrated_y.astype(float)-calibrated_y.astype(float).mean(0)
    before=(x-x.mean(0)).T@(y-y.mean(0))/n
    stats=dict(history_samples=int(memory.n),replay_samples=n,
        before_cross_error=float(np.linalg.norm(before-cross)/max(np.linalg.norm(cross),1e-12)),
        after_cross_error=float(np.linalg.norm(cx.T@cy/n-cross)/max(np.linalg.norm(cross),1e-12)),
        after_target_cov_error=float(np.linalg.norm(cy.T@cy/n-yy)/max(np.linalg.norm(yy),1e-12)),
        after_input_variance_error=float(np.linalg.norm(np.mean(cx**2,axis=0)-variance)/max(np.linalg.norm(variance),1e-12)))
    return calibrated_x,calibrated_y,stats

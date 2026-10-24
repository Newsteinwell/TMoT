"""Arrival-only unit-input moments anchor a changing joint representation."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from .fecam import unit_rows

def block_features(function,x):
    f=function.transform(x)
    return np.c_[unit_rows(f[:,:function.base.dimension]),unit_rows(f[:,function.base.dimension:])]

def weighted_stats(x,w,n):
    x=np.asarray(x,np.float64);w=np.asarray(w,np.float64);np.testing.assert_allclose(w.sum(),n,atol=1e-9)
    mu=np.sum(w[:,None]*x,axis=0)/n
    cov=(x-mu).T@(w[:,None]*(x-mu))/max(n-1,1)
    return mu,(cov+cov.T)/2

def complete_class(mean,cov,exact_mean,exact_cov,rho=.01):
    d=len(exact_mean);a=cov[:d,:d];u=cov[:d,d:];h=cov[d:,d:]
    scale=max(float(np.trace(a)/d),1e-12)
    b=cho_solve(cho_factor(a+rho*scale*np.eye(d),lower=True),u)
    residual=h-b.T@u-u.T@b+b.T@a@b;vals,vec=np.linalg.eigh((residual+residual.T)/2)
    remainder=(vec*np.maximum(vals,0))@vec.T
    result=np.block([[exact_cov,exact_cov@b],[b.T@exact_cov,b.T@exact_cov@b+remainder]])
    out_mean=np.r_[exact_mean,mean[d:]+(exact_mean-mean[:d])@b]
    return out_mean,result,dict(residual_min_eigenvalue=float(vals.min()),projection_norm=float(np.linalg.norm(b)),base_cov_error=float(np.max(abs(result[:d,:d]-exact_cov))),base_mean_error=float(np.max(abs(out_mean[:d]-exact_mean))))

def fast_precision(cov,alpha):
    d=len(cov);scale=max(float(np.trace(cov)/d),1e-12);reg=cov+alpha*scale*np.eye(d)
    sd=np.sqrt(np.maximum(np.diag(reg),1e-15));correlation=reg/np.outer(sd,sd)
    return cho_solve(cho_factor(correlation,lower=True),np.eye(d))

def statistics(learner,unit_memory):
    result={mode:[] for mode in ('base','approx','self','complete')};diagnostics=[]
    for label in learner.classes:
        exact=unit_memory.cells[label];mu=exact.total[:-1]/exact.n
        cov=(exact.gram[:-1,:-1]-exact.n*np.outer(mu,mu))/max(exact.n-1,1);cov=(cov+cov.T)/2
        cells=learner.local.cells[label];points=np.concatenate([c.points for c in cells]);w=np.concatenate([np.full(len(c.points),c.n/len(c.points)) for c in cells]);assert sum(c.n for c in cells)==exact.n
        joint_mean,joint_cov=weighted_stats(block_features(learner.function,points),w,exact.n)
        sm,sc,_=complete_class(joint_mean,joint_cov,joint_mean[:learner.dimension],joint_cov[:learner.dimension,:learner.dimension])
        cm,cc,diag=complete_class(joint_mean,joint_cov,mu,cov)
        result['base'].append((mu,cov));result['approx'].append((joint_mean,joint_cov));result['self'].append((sm,sc));result['complete'].append((cm,cc));diagnostics.append(dict(label=int(label),**diag))
    return result,diagnostics

def predict_stats(x,stats,alpha,classes):
    distances=[]
    for mu,cov in stats:
        p=fast_precision(cov,alpha);delta=x-mu
        distances.append(np.einsum('ni,ni->n',delta@p,delta))
    return np.asarray(classes)[np.column_stack(distances).argmin(1)]

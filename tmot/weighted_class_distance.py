"""Class-conditional correlation distance from weighted representative moments."""
import numpy as np
from .fecam import unit_rows

def class_statistics(x,weight,population):
    x=unit_rows(x);w=np.asarray(weight,np.float64)
    assert len(x)==len(w) and population>=1 and np.all(w>=0)
    np.testing.assert_allclose(w.sum(),population,atol=1e-9)
    mean=(w[:,None]*x).sum(0)/population
    cov=(x.T@(w[:,None]*x)-population*np.outer(mean,mean))/max(population-1,1)
    cov=(cov+cov.T)/2
    # Independent centered accumulation, before any shrinkage or inverse.
    centered=(x-mean).T@(w[:,None]*(x-mean))/max(population-1,1)
    error=float(np.max(abs(cov-centered)))
    np.testing.assert_allclose(cov,centered,rtol=1e-7,atol=1e-10)
    return mean,cov,error

def precision(cov,alpha):
    d=len(cov);scale=max(float(np.trace(cov)/d),1e-12)
    regularized=cov+alpha*scale*np.eye(d);sd=np.sqrt(np.maximum(np.diag(regularized),1e-15))
    correlation=regularized/np.outer(sd,sd)
    return np.linalg.pinv(correlation,hermitian=True)

def distances(x,mean,p):
    delta=unit_rows(x)-mean
    return np.einsum('ni,ni->n',delta@p,delta)

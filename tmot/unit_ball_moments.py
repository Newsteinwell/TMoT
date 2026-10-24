"""Contract an overflowing normalized-feature moment block by congruence."""
import numpy as np

def bound_hidden_moments(mean,cov,population,input_dimension):
    mean=np.asarray(mean,np.float64);cov=np.asarray(cov,np.float64)
    if population<1 or not 0<input_dimension<len(mean):raise ValueError('Invalid population or feature block')
    d=input_dimension;factor=max(population-1,1)/population
    before=float(mean[d:]@mean[d:]+factor*np.trace(cov[d:,d:]))
    if not np.isfinite(before) or before< -1e-10:raise ValueError('Invalid second moment')
    scale=min(1.,1/np.sqrt(max(before,1e-30)))
    mu=mean.copy();result=cov.copy();mu[d:]*=scale;result[:,d:]*=scale;result[d:,:]*=scale
    after=float(mu[d:]@mu[d:]+factor*np.trace(result[d:,d:]))
    assert after<=1+1e-12
    np.testing.assert_array_equal(mu[:d],mean[:d]);np.testing.assert_array_equal(result[:d,:d],cov[:d,:d])
    return mu,result,dict(second_norm_before=before,second_norm_after=after,contraction=float(scale),changed=bool(scale<1.))

def bound_statistics(stats,unit_memory,classes,input_dimension):
    result=[];diagnostics=[]
    for label,(mean,cov) in zip(classes,stats):
        mu,cc,diag=bound_hidden_moments(mean,cov,unit_memory.cells[label].n,input_dimension)
        result.append((mu,cc));diagnostics.append(dict(label=int(label),**diag))
    return result,diagnostics

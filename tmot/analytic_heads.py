"""Fixed-coordinate cumulative heads; no retained real examples."""
import numpy as np
from .moments import ridge_readout

def make_heads(memory,classes):
    means=np.asarray([memory.cells[c].total[:-1]/memory.cells[c].n for c in classes])
    within=sum((memory.cells[c].gram[:-1,:-1]-memory.cells[c].n*np.outer(mu,mu) for c,mu in zip(classes,means)),np.zeros((memory.dimension,memory.dimension)))
    n=sum(memory.cells[c].n for c in classes)
    covariance=(within+within.T)/(2*max(n-len(classes),1))
    scale=max(float(np.trace(covariance)/memory.dimension),1e-8)
    heads={'ncm':dict(kind='ncm',means=means)}
    for shrink in (.001,.01,.1,.5,.9,1.):
        precision=np.linalg.inv((1-shrink)*covariance+shrink*scale*np.eye(memory.dimension))
        heads[f'slda_{shrink:g}']=dict(kind='slda',means=means,precision=precision,shrink=shrink)
    g,k,_=memory.risk_moments(classes)
    for ridge in (.001,.01,.1,1.,10.,100.):
        heads[f'ridge_{ridge:g}']=dict(kind='ridge',weight=ridge_readout(g,k,ridge),ridge=ridge)
    return heads

def predict_head(head,x,classes):
    x=np.asarray(x,np.float64)
    if head['kind']=='ridge':scores=np.c_[x,np.ones(len(x))]@head['weight']
    else:
        means=head['means'];precision=head.get('precision',np.eye(x.shape[1]))
        scores=x@precision@means.T-.5*np.sum((means@precision)*means,axis=1)
    return np.asarray(classes)[scores.argmax(1)]

"""FeCAM head adapted to fixed time-series features.

Based on dipamgoswami/FeCAM, MIT, Copyright (c) 2023 Dipam Goswami.
See third_party/fecam for license, pinned source, and adaptation details.
"""
import numpy as np
from .moments import ClassificationMoments


def unit_rows(x):
    x=np.asarray(x,dtype=np.float64)
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


class FeCAMLearner:
    def __init__(self,dimension,alpha1=1.):
        if alpha1<=0:raise ValueError('Positive covariance shrinkage required')
        self.dimension=dimension;self.alpha1=alpha1
        self.memory=ClassificationMoments(dimension);self.classes=[]
        self.means={};self.precisions={};self.last_loss=None

    def observe(self,x,y):
        x=unit_rows(x);y=np.asarray(y)
        self.classes.extend(sorted(set(y.tolist())-set(self.classes)))
        self.memory.update(x,y)
        for c in np.unique(y):
            cell=self.memory.cells[c];mean=cell.total[:-1]/cell.n
            cov=(cell.gram[:-1,:-1]-cell.n*np.outer(mean,mean))/max(cell.n-1,1)
            cov=(cov+cov.T)/2
            scale=max(float(np.trace(cov)/self.dimension),1e-12)
            cov+=self.alpha1*scale*np.eye(self.dimension)
            sd=np.sqrt(np.maximum(np.diag(cov),1e-15))
            correlation=cov/np.outer(sd,sd)
            self.means[c]=mean
            self.precisions[c]=np.linalg.pinv(correlation,hermitian=True)

    def distances(self,x):
        x=unit_rows(x);dist=[]
        for c in self.classes:
            delta=x-self.means[c]
            dist.append(np.einsum('ni,ni->n',delta@self.precisions[c],delta))
        return np.column_stack(dist)

    def predict(self,x):
        return np.asarray(self.classes)[self.distances(x).argmin(1)]

    @property
    def persistent_bytes(self):
        return self.memory.nbytes+sum(a.nbytes for a in self.means.values())+sum(a.nbytes for a in self.precisions.values())

"""Exact affine moment transport for an appended raw-coordinate branch."""
import numpy as np
from .moments import ClassificationMoments,RegressionMoments,augment,ridge_readout


def affine_diagonal_moments(g,k,scale,shift):
    """Transform homogeneous moments without storing a dense transform matrix."""
    cross=scale*g[:-1,-1];constant=g[-1,-1]
    tg=g.copy();tk=k.copy()
    tg[:-1,:-1]=g[:-1,:-1]*np.outer(scale,scale)+np.outer(cross,shift)+np.outer(shift,cross)+constant*np.outer(shift,shift)
    tg[:-1,-1]=cross+constant*shift;tg[-1,:-1]=tg[:-1,-1]
    tk[:-1]=scale[:,None]*k[:-1]+shift[:,None]*k[-1]
    return tg,tk


class RawScaleTransport:
    def __init__(self,dimension,raw_start,task='classification',ridge=.1,transport=True):
        if not 0<=raw_start<dimension:raise ValueError('Invalid raw branch')
        self.dimension,self.raw_start,self.task,self.ridge,self.transport=dimension,raw_start,task,ridge,transport
        self.classes=[];self.memory=None;self.last_loss=None;self.stage=0;self.diagnostics=None

    def observe(self,x,y):
        y=np.asarray(y)
        if self.task=='classification':
            self.classes.extend(sorted(set(y.tolist())-set(self.classes)))
            if self.memory is None:self.memory=ClassificationMoments(self.dimension)
        else:
            if y.ndim==1:y=y[:,None]
            if self.memory is None:self.memory=RegressionMoments(self.dimension,y.shape[1])
        self.memory.update(x,y);g,k,_=self.memory.risk_moments(self.classes)
        self.scale=np.ones(self.dimension);self.shift=np.zeros(self.dimension)
        if self.transport:
            mean=g[:-1,-1];variance=np.maximum(np.diag(g)[:-1]-mean**2,0)
            raw_variance=variance[self.raw_start:];floor=max(float(np.median(raw_variance))*.01,1e-6)
            self.scale[self.raw_start:]=1/np.sqrt(np.maximum(raw_variance,floor))
            self.shift[self.raw_start:]=-mean[self.raw_start:]*self.scale[self.raw_start:]
        tg,tk=affine_diagonal_moments(g,k,self.scale,self.shift)
        self.readout=ridge_readout(tg,tk,self.ridge);self.stage+=1
        self.diagnostics=dict(raw_start=self.raw_start,scale_min=float(self.scale[self.raw_start:].min()),scale_max=float(self.scale[self.raw_start:].max()),arrivals=self.stage)

    def predict(self,x):
        scores=augment(np.asarray(x)*self.scale+self.shift)@self.readout
        return np.asarray(self.classes)[scores.argmax(1)] if self.task=='classification' else scores

    @property
    def persistent_bytes(self):return self.memory.nbytes+self.scale.nbytes+self.shift.nbytes+self.readout.nbytes

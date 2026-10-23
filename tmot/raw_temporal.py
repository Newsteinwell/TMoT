"""Preserve raw temporal coordinates via first-arrival affine calibration."""
import numpy as np
from .stable_residual import StableSemanticBasis


def raw_temporal(x, windows=True):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=3 or x.shape[-1]<1 or not np.isfinite(x).all():raise ValueError('Finite N,C,T required')
    if not windows:return x[:,:,-1].astype(np.float32)
    n,c,t=x.shape;edges=np.linspace(0,t,9).astype(int);parts=[]
    for start,end in zip(edges[:-1],edges[1:]):
        parts.append(x[:,:,start:end].mean(-1) if end>start else x[:,:,min(start,t-1)])
    return np.concatenate([np.stack(parts,axis=-1).reshape(n,-1),x[:,:,-1]],axis=1).astype(np.float32)


class RawCoordinateBasis:
    def __init__(self,dimension):self.dimension=dimension;self.fitted=False

    def fit(self,x):
        if self.fitted:raise RuntimeError('First-arrival calibration is immutable')
        x=np.asarray(x,np.float32)
        if x.shape[1]!=self.dimension:raise ValueError('Wrong raw descriptor width')
        self.mean=x.mean(0);sd=x.std(0);self.scale=np.maximum(sd,max(float(np.median(sd))*.01,1e-4))
        self.fitted=True;return self

    def transform(self,x):
        if not self.fitted:raise RuntimeError('Fit first')
        return ((np.asarray(x,np.float32)-self.mean)/self.scale).astype(np.float32)


class RawAppendedBasis:
    def __init__(self,amplitude_dim,raw_dim,dimension=512,seed=1701):
        self.amplitude_dim,self.raw_dim=amplitude_dim,raw_dim;self.dimension=dimension+raw_dim
        self.base=StableSemanticBasis(amplitude_dim,dimension,seed);self.raw=RawCoordinateBasis(raw_dim)

    def split(self,x):
        x=np.asarray(x,np.float32);end=x.shape[1]-self.amplitude_dim;start=end-self.raw_dim
        if start<=0:raise ValueError('Missing semantic block')
        return np.concatenate([x[:,:start],x[:,end:]],axis=1),x[:,start:end]

    def fit(self,x):
        old,raw=self.split(x);self.base.fit(old);self.raw.fit(raw);return self

    def transform(self,x):
        old,raw=self.split(x)
        return np.concatenate([self.base.transform(old),self.raw.transform(raw)],axis=1)

"""Sample-local temporal/cross-channel descriptors and first-arrival fusion."""
import numpy as np
from .stable_residual import StableSemanticBasis


def descriptor_dimension(channels):
    return 32*channels+channels*(channels-1)//2+3*channels*channels


def temporal_descriptor(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=3 or x.shape[-1]<2 or not np.isfinite(x).all():raise ValueError('Finite N,C,T input required with T>=2')
    n,c,t=x.shape;mean=x.mean(-1);sd=x.std(-1);z=(x-mean[:,:,None])/np.maximum(sd[:,:,None],1e-6)
    # Fixed 32-point half-pixel resampling, retaining order within each channel.
    points=np.clip((np.arange(32)+.5)*t/32-.5,0,t-1);lo=np.floor(points).astype(int);hi=np.minimum(lo+1,t-1);fraction=points-lo
    shape=z[:,:,lo]*(1-fraction)+z[:,:,hi]*fraction
    zero=np.einsum('nct,ndt->ncd',z,z)/t;upper=np.triu_indices(c,k=1)
    parts=[shape.reshape(n,-1),zero[:,upper[0],upper[1]]]
    for requested in (1,2,4):
        lag=min(requested,t-1)
        parts.append((np.einsum('nct,ndt->ncd',z[:,:,:-lag],z[:,:,lag:])/(t-lag)).reshape(n,-1))
    descriptor=np.concatenate(parts,axis=1).astype(np.float32)
    assert descriptor.shape==(n,descriptor_dimension(c))
    return descriptor


class JointTemporalBasis:
    def __init__(self,amplitude_dim,joint_dim,dimension=512,seed=1701,strength=1.):
        self.amplitude_dim,self.joint_dim,self.dimension,self.seed,self.strength=amplitude_dim,joint_dim,dimension,seed,strength
        self.base=StableSemanticBasis(amplitude_dim,dimension,seed);self.fitted=False

    def split(self,x):
        x=np.asarray(x,np.float32);end=x.shape[1]-self.amplitude_dim;start=end-self.joint_dim
        if start<=0:raise ValueError('Missing Mantis semantic block')
        return np.concatenate([x[:,:start],x[:,end:]],axis=1),x[:,start:end]

    def fit(self,x):
        if self.fitted:raise RuntimeError('First-stage calibration is immutable')
        original,joint=self.split(x);self.base.fit(original)
        self.mean=joint.mean(0);self.scale=max(float(np.sqrt(np.mean((joint-self.mean)**2))),1e-6)
        width=self.dimension-self.base.amp_width
        self.projection=(np.random.default_rng(self.seed+101).normal(size=(self.joint_dim,width))/np.sqrt(self.joint_dim)).astype(np.float32)
        self.fitted=True;return self

    def transform(self,x):
        if not self.fitted:raise RuntimeError('Fit first')
        original,joint=self.split(x);z=self.base.transform(original)
        if self.strength==0:return z
        width=self.dimension-self.base.amp_width
        addition=((joint-self.mean)/self.scale)@self.projection
        z[:,:width]=(z[:,:width]+self.strength*addition)/np.sqrt(1+self.strength**2)
        return z.astype(np.float32)

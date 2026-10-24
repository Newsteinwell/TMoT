"""Sample-local lagged channel products, calibrated only on the first arrival."""
import numpy as np
from .stable_residual import StableSemanticBasis

def lagged_products(x):
    x=np.asarray(x,np.float64)
    if x.ndim!=3 or x.shape[-1]<5 or not np.isfinite(x).all():
        raise ValueError('Expected finite N,C,T observations with T>=5')
    z=x-x.mean(-1,keepdims=True)
    z=z/np.maximum(np.sqrt(np.mean(z*z,axis=-1,keepdims=True)),1e-6)
    blocks=[]
    for lag in (0,1,4):
        left=z if not lag else z[:,:,:-lag]
        right=z if not lag else z[:,:,lag:]
        blocks.append((np.einsum('nct,ndt->ncd',left,right)/left.shape[-1]).reshape(len(x),-1))
    return np.concatenate(blocks,axis=1).astype(np.float32)

class InteractionBasis:
    def __init__(self,channels,enabled):
        self.base=StableSemanticBasis(2*channels,512);self.enabled=enabled
        self.dimension=576 if enabled else 512
    def fit(self,features,raw):
        self.base.fit(features)
        if self.enabled:
            h=lagged_products(raw);self.mean=h.mean(0)
            self.scale=max(float(np.sqrt(np.mean((h-self.mean)**2))),1e-4)
            self.projection=(np.random.default_rng(2718).normal(size=(h.shape[1],64))/np.sqrt(h.shape[1])).astype(np.float32)
            # Match per-coordinate RMS, not total block energy: only 64 new coordinates.
            self.output_scale=max(float(np.sqrt(np.mean(self.base.transform(features)**2))),1e-4)
        return self
    def transform(self,features,raw):
        z=self.base.transform(features)
        if not self.enabled:return z
        h=((lagged_products(raw)-self.mean)/self.scale)@self.projection*self.output_scale
        return np.concatenate([z,h],axis=1).astype(np.float32)

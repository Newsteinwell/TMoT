"""Nonlinear fixed-basis moments with online affine variance transport.

Linear features are retained; fixed random Fourier features add nonlinear
capacity without invalidating exact historical moment transport. Online scaling
uses only arrived aggregate moments. All options are experimental candidates.
"""
import numpy as np
from .stable_residual import StableSemanticBasis
from .moments import ClassificationMoments,RegressionMoments,augment,transport_risk,ridge_readout

class KernelBasis:
    def __init__(self,amplitude_dim,dimension=256,seed=1701):
        self.linear=StableSemanticBasis(amplitude_dim,dimension,seed)
        self.dimension=dimension*3;self.seed=seed

    def fit(self,x):
        self.linear.fit(x);z=self.linear.transform(x)
        # Fit only a global distance scale; no supervised first-class truncation.
        self.scale=max(float(np.sqrt(np.mean(z*z))),1e-6)
        rng=np.random.default_rng(self.seed+97);d=self.linear.dimension
        self.frequencies=(rng.normal(size=(d,2*d))/np.sqrt(d)).astype(np.float32)
        # Fixed two length scales, not selected separately per test dataset.
        self.frequencies[:,:d]*=.5;self.frequencies[:,d:]*=2.
        self.phase=rng.uniform(0,2*np.pi,size=2*d).astype(np.float32)
        return self

    def transform(self,x):
        z=self.linear.transform(x)/self.scale
        nonlin=np.sqrt(2.)*np.cos(z@self.frequencies+self.phase)
        return np.concatenate([z,nonlin],axis=1).astype(np.float32)

class KernelTransportLearner:
    def __init__(self,dimension,task='classification',ridge=.1,transport=True):
        self.dimension,self.task,self.ridge,self.transport=dimension,task,ridge,transport
        self.classes=[];self.memory=None;self.last_loss=None;self.stage=0

    def observe(self,x,y):
        y=np.asarray(y)
        if self.task=='classification':
            self.classes.extend(sorted(set(y.tolist())-set(self.classes)))
            if self.memory is None:self.memory=ClassificationMoments(self.dimension)
        else:
            if y.ndim==1:y=y[:,None]
            if self.memory is None:self.memory=RegressionMoments(self.dimension,y.shape[1])
        self.memory.update(x,y);g,k,_=self.memory.risk_moments(self.classes)
        self.h=np.eye(self.dimension+1)
        if self.transport:
            mean=g[:-1,-1];variance=np.maximum(np.diag(g)[:-1]-mean**2,0)
            floor=max(float(np.median(variance))*.01,1e-6);sd=np.sqrt(np.maximum(variance,floor))
            self.h[:-1,:-1]=np.diag(1/sd);self.h[:-1,-1]=-mean/sd
        tg,tk=transport_risk(g,k,self.h)
        self.readout=ridge_readout(tg,tk,self.ridge);self.stage+=1

    def predict(self,x):
        scores=augment(x)@self.h.T@self.readout
        return np.asarray(self.classes)[scores.argmax(1)] if self.task=='classification' else scores

    @property
    def persistent_bytes(self):return self.memory.nbytes+self.h.nbytes+self.readout.nbytes

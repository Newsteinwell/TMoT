"""Adaptive functions with weighted replay; nonlinear history risk is approximate."""
import numpy as np
from .functional_basis import FunctionalBasis
from .local_moments import LocalMomentMixture
from .gradient_diagnostic import empirical_sample
from .nonlinear_moments import NonlinearMomentLearner
from .moments import augment,ridge_readout

class IdentityBasis:
    def __init__(self,dimension):self.dimension=dimension
    def transform(self,x):return np.asarray(x,np.float32)

def representative_weights(current,history,task,counts,seen):
    n=len(current);h=0 if history is None else len(history)
    if task=='regression':
        return np.r_[np.full(n,1/(seen+n)),np.full(h,seen/((seen+n)*h)) if h else np.empty(0)]
    classes=np.unique(current if history is None else np.r_[current,history]);w=np.zeros(n+h)
    for c in classes:
        here=np.flatnonzero(current==c);old=np.flatnonzero(history==c) if h else np.empty(0,dtype=int)
        previous=counts.get(int(c),0);total=previous+len(here)
        if previous and not len(old):raise ValueError('Historical class absent from representatives')
        w[here]=1/(len(classes)*total)
        if len(old):w[n+old]=previous/(len(classes)*total*len(old))
    return w

class ReplayFunctionalLearner:
    def __init__(self,dimension,task='classification',seed=42,ridge=.1,mode='local',epochs=30,device='cuda',replay_count=4096,replay_bytes=4*1024**2):
        if mode not in ('local','er'):raise ValueError(mode)
        self.dimension=dimension;self.task=task;self.seed=seed;self.ridge=ridge;self.mode=mode;self.epochs=epochs;self.device=device
        self.replay_count=replay_count;self.replay_bytes=replay_bytes;self.rng=np.random.default_rng(seed)
        self.local=LocalMomentMixture(dimension,task,seed+676,components=4,preserve_targets=True,support=4,min_cell=4) if mode=='local' else None
        self.buffer=None;self.keys=None;self.classes=[];self.counts={};self.seen=0;self.stage=0;self.last_loss=None
        self.last_calibration=None;self.warm_init_error=None;self.input_stats=None
    def observe(self,x,y):
        x=np.asarray(x,np.float32);y=np.asarray(y)
        if self.task=='regression' and y.ndim==1:y=y[:,None]
        historical=None
        if self.seen:
            historical=self.local.sample(self.replay_count,self.classes,self.rng) if self.mode=='local' else self.sample_buffer()
        hx,hy=historical if historical is not None else (None,None)
        tx=x if hx is None else np.r_[x,hx];ty=y if hy is None else np.r_[y,hy]
        w=self.replay_weights(y,hy)
        np.testing.assert_allclose(w.sum(),1,atol=1e-12)
        self.function=FunctionalBasis(IdentityBasis(self.dimension),self.task,seed=self.seed,epochs=self.epochs,device=self.device).fit(tx,ty,sample_weight=w,input_stats=self.input_stats)
        if self.input_stats is None:self.input_stats=(self.function.mean.copy(),self.function.sd.copy())
        if self.task=='classification':
            self.classes.extend(sorted(set(y.tolist())-set(self.classes)))
            target=np.eye(len(self.classes))[[self.classes.index(int(c)) for c in ty]]
        else:target=ty
        z=augment(self.function.transform(tx));self.g=z.T@(w[:,None]*z);self.k=z.T@(w[:,None]*target)
        self.readout=ridge_readout(self.g,self.k,self.ridge)
        self.optimizer_steps=self.function.optimizer_steps;self.last_loss=self.function.last_loss
        self.historical_training_samples=0 if hx is None else len(hx)
        self.weight_summary=dict(current=float(w[:len(x)].sum()),history=float(w[len(x):].sum()),total=float(w.sum()))
        if self.mode=='local':self.update_local(x,y)
        else:self.store_buffer(x,y)
        if self.task=='classification':
            for c in np.unique(y):self.counts[int(c)]=self.counts.get(int(c),0)+int(np.sum(y==c))
        self.seen+=len(x);self.stage+=1
    def sample_buffer(self):
        return empirical_sample(*self.buffer,self.replay_count,self.rng,self.classes)
    def replay_weights(self,y,hy):
        return representative_weights(y,hy,self.task,self.counts,self.seen)
    def store_buffer(self,x,y):
        NonlinearMomentLearner._store_buffer(self,x,y)
    def update_local(self,x,y):
        self.local.update(x,y)
    def predict(self,x):
        scores=augment(self.function.transform(x))@self.readout
        return np.asarray(self.classes)[scores.argmax(1)] if self.task=='classification' else scores
    @property
    def persistent_bytes(self):
        size=self.g.nbytes+self.k.nbytes+self.readout.nbytes+sum(v.nbytes for v in self.input_stats)
        size+=self.function.mean.nbytes+self.function.sd.nbytes+self.function.hidden_mean.nbytes+8
        size+=sum(p.numel()*p.element_size() for p in self.function.net.parameters())
        if self.local is not None:size+=self.local.nbytes
        if self.buffer is not None:size+=sum(v.nbytes for v in self.buffer)+self.keys.nbytes
        return size

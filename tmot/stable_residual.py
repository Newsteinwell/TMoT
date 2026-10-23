"""Stable full-basis readout plus a low-rank moment-constrained score residual.

A fixed semantic branch avoids first-stage supervised rank truncation. The
cumulative analytic predictor is retained explicitly; adaptation starts from zero
score correction at each arrival. Exact historical quadratic risk remains valid.
This is a candidate, not a claim of universal superiority.
"""
import numpy as np
import torch
from torch import nn
from .moments import ClassificationMoments,RegressionMoments,augment,quadratic_risk,ridge_readout

class StableSemanticBasis:
    def __init__(self,amplitude_dim,dimension=512,seed=1701):
        self.amplitude_dim,self.dimension,self.seed=amplitude_dim,dimension,seed;self.fitted=False

    def fit(self,x):
        if self.fitted:raise RuntimeError('Immutable first-stage calibration')
        x=np.asarray(x,np.float32);semantic=x.shape[1]-self.amplitude_dim
        if semantic<=0 or len(x)<2:raise ValueError('Invalid basis input')
        self.semantic_dim=semantic
        # Shared semantic RMS, not per-coordinate variance from a few initial classes.
        self.mean=x[:,:semantic].mean(0);self.scale=max(float(np.sqrt(np.mean((x[:,:semantic]-self.mean)**2))),1e-6)
        amp=x[:,semantic:];self.amp_mean=amp.mean(0);sd=amp.std(0)
        self.amp_sd=np.maximum(sd,max(float(np.median(sd))*.1,1e-4))
        rng=np.random.default_rng(self.seed);self.amp_width=min(64,self.dimension//4)
        self.semantic_projection=(rng.normal(size=(semantic,self.dimension-self.amp_width))/np.sqrt(semantic)).astype(np.float32)
        self.amp_projection=(rng.normal(size=(self.amplitude_dim,self.amp_width))/np.sqrt(self.amplitude_dim)).astype(np.float32)
        self.fitted=True;return self

    def transform(self,x):
        if not self.fitted:raise RuntimeError('Fit first')
        x=np.asarray(x,np.float32)
        sem=((x[:,:self.semantic_dim]-self.mean)/self.scale)@self.semantic_projection
        amp=((x[:,self.semantic_dim:]-self.amp_mean)/self.amp_sd)@self.amp_projection
        return np.concatenate([sem,amp],axis=1).astype(np.float32)

class StableResidualLearner:
    def __init__(self,dimension,task='classification',seed=42,rank=64,epochs=30,ridge=.1,
                 hist_weight=1.,residual_weight=.1,alpha=1.,device='cuda'):
        self.dimension,self.task,self.seed,self.rank=dimension,task,seed,min(rank,dimension)
        self.epochs,self.ridge,self.hist_weight,self.residual_weight,self.alpha=epochs,ridge,hist_weight,residual_weight,alpha
        self.device=device;self.classes=[];self.memory=None;self.stage=0;self.last_loss=None

    def observe(self,x,y):
        x=np.asarray(x,np.float32);y=np.asarray(y);torch.manual_seed(self.seed+self.stage*100)
        rng=np.random.default_rng(self.seed+self.stage*100)
        if self.task=='classification':
            self.classes.extend(sorted(set(y.tolist())-set(self.classes)));outputs=len(self.classes)
            local=np.array([self.classes.index(c) for c in y]);memory_type=ClassificationMoments
            if self.memory is None:self.memory=memory_type(self.dimension)
        else:
            if y.ndim==1:y=y[:,None]
            local=y.astype(np.float32);outputs=y.shape[1]
            if self.memory is None:self.memory=RegressionMoments(self.dimension,outputs)
        # Build historical constraints before current rows enter persistent sufficient statistics.
        old=None
        if self.stage:old=[torch.as_tensor(v,dtype=torch.float64,device=self.device) for v in self.memory.risk_moments(self.classes)]
        self.memory.update(x,y);g,k,v=self.memory.risk_moments(self.classes)
        self.base=ridge_readout(g,k,self.ridge)
        self.adapter=nn.Linear(self.dimension,self.rank,bias=False).to(self.device)
        self.correction=nn.Linear(self.rank,outputs).to(self.device)
        with torch.no_grad():
            self.adapter.weight.copy_(torch.eye(self.dimension)[:self.rank]);self.correction.weight.zero_();self.correction.bias.zero_()
        tx=torch.as_tensor(x,device=self.device);ty=torch.as_tensor(local,device=self.device)
        base=torch.as_tensor(self.base.T,dtype=torch.float64,device=self.device)
        opt=torch.optim.AdamW(list(self.adapter.parameters())+list(self.correction.parameters()),lr=.001,weight_decay=1e-4)
        losses=[]
        for epoch in range(self.epochs if self.alpha else 0):
            order=rng.permutation(len(x))
            for start in range(0,len(x),128):
                ids=order[start:start+128]
                correction=self.correction(self.adapter(tx[ids]));z=torch.cat([tx[ids],torch.ones((len(ids),1),device=self.device)],1)
                scores=z.double()@base.T+self.alpha*correction.double()
                loss=(nn.functional.cross_entropy(scores,ty[ids]) if self.task=='classification' else (scores-ty[ids]).square().sum(1).mean())
                delta=torch.cat([self.correction.weight@self.adapter.weight,self.correction.bias[:,None]],1).double()
                f=base+self.alpha*delta
                if old is not None:loss=loss+self.hist_weight*quadratic_risk(f,*old)
                loss=loss+self.residual_weight*delta.square().sum()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite residual objective')
                opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
        self.last_loss=float(np.mean(losses[-int(np.ceil(len(x)/128)):])) if losses else None
        self.stage+=1

    def predict(self,x):
        x=np.asarray(x,np.float32)
        with torch.no_grad():correction=self.correction(self.adapter(torch.as_tensor(x,device=self.device))).cpu().numpy()
        scores=augment(x)@self.base+self.alpha*correction
        return np.asarray(self.classes)[scores.argmax(1)] if self.task=='classification' else scores

    @property
    def persistent_bytes(self):
        total=self.memory.nbytes+self.base.nbytes
        total+=sum(p.numel()*p.element_size() for m in (self.adapter,self.correction) for p in m.parameters())
        return total

"""Nonlinear residual with approximate moment replay, not exact nonlinear risk.

The analytic shortcut retains exact fixed-basis moments. Gaussian replay uses
class-conditional moments or per-arrival joint input/target moments. No claim
that second moments determine nonlinear losses is made.
"""
import numpy as np
import torch
from torch import nn
from .moments import Cell, ClassificationMoments, RegressionMoments, augment, ridge_readout


def sample_cell(cell, count, rng, shrinkage=.1):
    mean = cell.total[:-1] / cell.n
    cov = cell.gram[:-1, :-1] / cell.n - np.outer(mean, mean)
    cov = (cov + cov.T) / 2
    cov = (1-shrinkage)*cov + shrinkage*np.diag(np.maximum(np.diag(cov), 0))
    eig, vec = np.linalg.eigh(cov)
    factor = vec * np.sqrt(np.maximum(eig, 0))
    return (mean + rng.normal(size=(count, len(mean))) @ factor.T).astype(np.float32)


class NonlinearMomentLearner:
    def __init__(self, dimension, task='classification', seed=42, ridge=.1,
                 mode='gaussian', epochs=30, residual_weight=.1, device='cuda',
                 replay_count=4096, replay_bytes=4*1024**2, width=128, alpha=1., er_resample=False, local_components=4):
        if mode not in ('gaussian', 'local', 'er', 'no_history'): raise ValueError(mode)
        self.dimension, self.task, self.seed = dimension, task, seed
        self.ridge, self.mode, self.epochs = ridge, mode, epochs
        self.residual_weight, self.device = residual_weight, device
        self.replay_count, self.replay_bytes, self.width, self.alpha = replay_count, replay_bytes, width, alpha
        self.er_resample=er_resample
        from .local_moments import LocalMomentMixture
        self.local=LocalMomentMixture(dimension,task,seed+676,components=local_components) if mode=='local' else None
        self.classes=[]; self.memory=None; self.joint_cells=[]; self.stage=0; self.seen=0
        self.buffer=None; self.keys=None; self.last_loss=None
        self.rng=np.random.default_rng(seed)

    def _historical(self, outputs):
        if not self.stage or self.mode=='no_history': return None
        if self.mode=='local':return self.local.sample(self.replay_count,self.classes,self.rng)
        if self.mode=='er':
            x,y=self.buffer
            count=self.replay_count if self.er_resample else min(len(x),self.replay_count)
            ids=self.rng.choice(len(x), size=count, replace=count>len(x))
            return x[ids].copy(), y[ids].copy()
        if self.task=='classification':
            # Balanced class sampling matches the macro-risk shortcut objective.
            counts=self.rng.multinomial(self.replay_count, np.ones(len(self.classes))/len(self.classes))
            xs=[];ys=[]
            for c,n in zip(self.classes,counts):
                if n:
                    xs.append(sample_cell(self.memory.cells[c],n,self.rng));ys.append(np.full(n,c))
            return np.concatenate(xs),np.concatenate(ys)
        weights=np.array([cell.n for cell in self.joint_cells],dtype=float);weights/=weights.sum()
        counts=self.rng.multinomial(self.replay_count,weights)
        joint=np.concatenate([sample_cell(cell,n,self.rng) for cell,n in zip(self.joint_cells,counts) if n])
        return joint[:,:self.dimension],joint[:,self.dimension:]

    def _store_buffer(self,x,y):
        keys=self.rng.random(len(x))
        if self.buffer is not None:
            x=np.concatenate([self.buffer[0],x]);y=np.concatenate([self.buffer[1],y]);keys=np.r_[self.keys,keys]
        row_bytes=x[0].nbytes+np.asarray(y[0]).nbytes+8
        keep=min(len(x),max(1,self.replay_bytes//row_bytes))
        ids=np.argsort(keys)[-keep:]
        self.buffer=(x[ids].copy(),y[ids].copy());self.keys=keys[ids].copy()

    def observe(self,x,y):
        x=np.asarray(x,np.float32);y=np.asarray(y)
        if self.task=='regression' and y.ndim==1:y=y[:,None]
        # Snapshot historical distributions before adding any current observations.
        historical=self._historical(None)
        if self.task=='classification':
            self.classes.extend(sorted(set(y.tolist())-set(self.classes)));outputs=len(self.classes)
            if self.memory is None:self.memory=ClassificationMoments(self.dimension)
        else:
            outputs=y.shape[1]
            if self.memory is None:self.memory=RegressionMoments(self.dimension,outputs)
        self.memory.update(x,y);g,k,_=self.memory.risk_moments(self.classes)
        self.base=ridge_readout(g,k,self.ridge)
        self.mean=g[:-1,-1].astype(np.float32)
        var=np.maximum(np.diag(g)[:-1]-self.mean.astype(float)**2,0)
        self.sd=np.sqrt(np.maximum(var,max(np.median(var)*.01,1e-6))).astype(np.float32)
        if historical is None:tx,ty=x,y
        else:tx=np.concatenate([x,historical[0]]);ty=np.concatenate([y,historical[1]])
        if self.task=='classification':
            ids=np.array([self.classes.index(c) for c in ty]);targets=np.eye(outputs,dtype=np.float32)[ids]
            counts=np.bincount(ids,minlength=outputs)
            weight=1./np.maximum(counts[ids],1);weight/=weight.sum()
        else:
            targets=ty.astype(np.float32)
            weight=np.ones(len(tx))/len(tx)
            if historical is not None:
                weight[:len(x)]=1./(self.seen+len(x))
                weight[len(x):]=self.seen/(self.seen+len(x))/len(historical[0])
        torch.manual_seed(self.seed+self.stage*100)
        self.net=nn.Sequential(nn.Linear(self.dimension,self.width),nn.ReLU(),
                               nn.Linear(self.width,self.width),nn.ReLU(),nn.Linear(self.width,outputs)).to(self.device)
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        optimizer=torch.optim.AdamW(self.net.parameters(),lr=.001,weight_decay=1e-4)
        features=torch.as_tensor((tx-self.mean)/self.sd,device=self.device)
        base=torch.as_tensor(augment(tx)@self.base,dtype=torch.float32,device=self.device)
        targets=torch.as_tensor(targets,device=self.device)
        weight=torch.as_tensor(weight*len(tx),dtype=torch.float32,device=self.device)
        self.net.train()
        self.historical_training_samples=0 if historical is None else len(historical[0])
        self.optimizer_steps=0
        for epoch in range(self.epochs):
            order=torch.randperm(len(tx),device=self.device)
            for batch in order.split(256):
                delta=self.net(features[batch]);pred=base[batch]+self.alpha*delta
                error=(pred-targets[batch]).square().mean(1)
                loss=(weight[batch]*(error+self.residual_weight*delta.square().mean(1))).mean()
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.net.parameters(),5.);optimizer.step()
                self.optimizer_steps+=1
                self.last_loss=float(loss.detach())
        self.net.eval()
        if self.mode=='local':self.local.update(x,y)
        elif self.mode=='er':self._store_buffer(x,y)
        elif self.mode=='gaussian' and self.task=='regression':self.joint_cells.append(Cell.from_x(np.concatenate([x,y],axis=1)))
        self.stage+=1;self.seen+=len(x)

    def predict(self,x):
        x=np.asarray(x,np.float32)
        with torch.no_grad():
            delta=self.net(torch.as_tensor((x-self.mean)/self.sd,device=self.device)).cpu().numpy()
        scores=augment(x)@self.base+self.alpha*delta
        return np.asarray(self.classes)[scores.argmax(1)] if self.task=='classification' else scores

    @property
    def persistent_bytes(self):
        size=self.memory.nbytes+self.base.nbytes+self.mean.nbytes+self.sd.nbytes
        size+=sum(p.numel()*p.element_size() for p in self.net.parameters())
        size+=sum(c.total.nbytes+c.gram.nbytes+8 for c in self.joint_cells)
        if self.local is not None:size+=self.local.nbytes
        if self.buffer is not None:size+=sum(v.nbytes for v in self.buffer)+self.keys.nbytes
        return size

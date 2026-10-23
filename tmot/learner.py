"""Pilot learners. TMoT is a new generalization, NOT a faithful WiMoT reproduction."""
import numpy as np
import torch
from torch import nn
from .moments import (ClassificationMoments, RegressionMoments, augment,
                      affine_matrix, transport_risk, quadratic_risk, tensor_moments, ridge_readout)


class FixedBasis:
    """First-stage-only scaling followed by fixed random nonlinear projection."""
    def __init__(self, dimension=256, seed=1701):
        self.dimension, self.seed = dimension, seed

    def fit(self, x):
        self.mean = x.mean(0)
        self.sd = np.maximum(x.std(0), max(float(np.median(x.std(0)))*.01,1e-5))
        rng=np.random.default_rng(self.seed)
        self.projection = (rng.normal(size=(x.shape[1],self.dimension))/np.sqrt(x.shape[1])).astype('float32')
        return self

    def transform(self, x):
        return np.maximum(((x-self.mean)/self.sd)@self.projection,0).astype('float32')


def supcon(h, y, temperature=.1):
    if len(h)<2: return h.sum()*0
    z=nn.functional.normalize(h,dim=1)
    logits=z@z.T/temperature
    eye=torch.eye(len(z),dtype=torch.bool,device=z.device)
    logits=logits.masked_fill(eye,-1e9)
    positives=(y[:,None]==y[None,:]) & ~eye
    valid=positives.sum(1)>0
    if not valid.any(): return h.sum()*0
    logp=logits-torch.logsumexp(logits,dim=1,keepdim=True)
    return -((logp*positives).sum(1)/positives.sum(1).clamp_min(1))[valid].mean()


class MomentLearner:
    def __init__(self, dimension, task='classification', method='tmot', seed=42,
                 bottleneck=64, epochs=30, lr=.001, hist_weight=1., sc_weight=.5,
                 ridge=.01, device='cpu', replay_bytes=4*1024**2,
                 readout_mode='ridge', current_loss='ce'):
        if method not in ('tmot','no_hist','frozen_adapter','ft','er','ridge','ranpac','slda'):
            raise ValueError(method)
        if task not in ('classification','regression'): raise ValueError(task)
        if task=='regression' and method=='slda': raise ValueError('SLDA is classification-only')
        self.dimension,self.task,self.method,self.seed=dimension,task,method,seed
        self.epochs,self.lr,self.hist_weight,self.sc_weight=epochs,lr,hist_weight,sc_weight
        if readout_mode not in ('ridge','trained'): raise ValueError(readout_mode)
        if current_loss not in ('ce','mse'): raise ValueError(current_loss)
        self.readout_mode,self.current_loss=readout_mode,current_loss
        self.ridge,self.device,self.replay_bytes=ridge,device,replay_bytes
        self.rng=np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.adapter=nn.Linear(dimension,min(bottleneck,dimension)).to(device)
        with torch.no_grad():
            self.adapter.weight.copy_(torch.eye(dimension)[:self.adapter.out_features]);self.adapter.bias.zero_()
        self.head=None; self.classes=[];self.memory=None;self.stage=0;self.buffer=None;self.keys=None
        self.rp=None
        if method=='ranpac':
            self.rp=(self.rng.normal(size=(dimension,512))/np.sqrt(dimension)).astype('float32')
        self.last_loss=None

    def features(self,x):
        return np.maximum(x@self.rp,0) if self.rp is not None else x

    def _expand(self,outputs):
        if self.head is not None and self.head.out_features==outputs:return
        old=self.head
        self.head=nn.Linear(self.adapter.out_features,outputs).to(self.device)
        if old is not None:
            with torch.no_grad():
                self.head.weight[:old.out_features].copy_(old.weight)
                self.head.bias[:old.out_features].copy_(old.bias)

    def observe(self,x,y):
        x=np.asarray(x,dtype='float32');y=np.asarray(y)
        if self.task=='classification':
            for c in sorted(set(y.tolist())-set(self.classes)):self.classes.append(int(c))
            outputs=len(self.classes)
            local=np.array([self.classes.index(int(c)) for c in y])
        else:
            if y.ndim==1:y=y[:,None]
            outputs=y.shape[1];local=y.astype('float32')
        xp=self.features(x)
        if self.memory is None:
            self.memory=(ClassificationMoments(xp.shape[1]) if self.task=='classification' else RegressionMoments(xp.shape[1],outputs))
        analytic=self.method in ('ridge','ranpac','slda')
        if not analytic:
            self._expand(outputs)
            old_moments = (tensor_moments(self.memory,self.classes,self.device)
                           if self.stage and self.method in ('tmot','frozen_adapter') else None)
            train_x,train_y=x,local
            if self.method=='er' and self.buffer is not None:
                bx,by=self.buffer
                if self.task=='classification':by=np.array([self.classes.index(int(c)) for c in by])
                train_x=np.concatenate([x,bx]);train_y=np.concatenate([local,by])
            tx=torch.as_tensor(train_x,device=self.device)
            ty=torch.as_tensor(train_y,device=self.device)
            opt=torch.optim.AdamW(list(self.adapter.parameters())+list(self.head.parameters()),lr=self.lr,weight_decay=1e-4)
            losses=[]
            for epoch in range(self.epochs if self.method!='frozen_adapter' or not self.stage else 0):
                order=self.rng.permutation(len(tx))
                for start in range(0,len(tx),128):
                    ids=order[start:start+128];h=self.adapter(tx[ids]);scores=self.head(h)
                    if self.task=='classification':
                        loss=(nn.functional.cross_entropy(scores,ty[ids]) if self.current_loss=='ce' else
                              (scores-nn.functional.one_hot(ty[ids],outputs)).square().sum(1).mean())
                        if self.method in ('tmot','no_hist','frozen_adapter'):
                            loss=loss+self.sc_weight*supcon(h,ty[ids])
                    else:loss=(scores-ty[ids]).square().sum(1).mean()
                    if old_moments is not None:
                        f=torch.cat([self.head.weight@self.adapter.weight,
                           (self.head.weight@self.adapter.bias+self.head.bias)[:,None]],dim=1).double()
                        loss=loss+self.hist_weight*quadratic_risk(f,*old_moments)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                    opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
            self.last_loss=float(np.mean(losses[-max(1,int(np.ceil(len(tx)/128))):])) if losses else None
        if self.method not in ('ft','er'):
            self.memory.update(xp,y)
            g,k,_=self.memory.risk_moments(self.classes)
            if self.method in ('tmot','no_hist','frozen_adapter'):
                h=affine_matrix(self.adapter.weight.detach().cpu().numpy(),self.adapter.bias.detach().cpu().numpy())
                g,k=transport_risk(g,k,h)
            if self.method=='slda':self._fit_lda()
            else:self.readout=ridge_readout(g,k,self.ridge)
        if self.method=='er':
            # Uniform random-priority reservoir; exact feature+label+priority byte accounting.
            bx,by=(self.buffer if self.buffer is not None else (x[:0],y[:0]))
            old_keys=self.keys if self.keys is not None else np.empty(0)
            bx=np.concatenate([bx,x]);by=np.concatenate([by,y]);keys=np.r_[old_keys,self.rng.random(len(x))]
            per_row=x.shape[1]*4+(8 if self.task=='classification' else outputs*8)+8
            cap=self.replay_bytes//per_row
            keep=np.argsort(keys)[:cap];self.buffer=(bx[keep],by[keep]);self.keys=keys[keep]
        self.stage+=1

    def _fit_lda(self):
        means=[];within=np.zeros((self.dimension,self.dimension));den=0
        for c in self.classes:
            cell=self.memory.cells[c];mu=cell.total[:-1]/cell.n;means.append(mu)
            within+=cell.gram[:-1,:-1]-cell.n*np.outer(mu,mu);den+=cell.n
        cov=within/max(den-len(means),1);scale=max(np.trace(cov)/len(cov),1e-8)
        cov=.9*cov+.1*scale*np.eye(len(cov))
        self.lda_means=np.array(means);self.lda_precision=np.linalg.inv(cov)

    def predict(self,x):
        x=np.asarray(x,dtype='float32')
        if self.method in ('ft','er','tmot','no_hist','frozen_adapter'):
            with torch.no_grad():
                h=self.adapter(torch.as_tensor(x,device=self.device))
                scores=self.head(h).cpu().numpy() if self.method in ('ft','er') or getattr(self,'readout_mode','ridge')=='trained' else augment(h.cpu().numpy())@self.readout
        elif self.method=='slda':
            m=self.lda_means;p=self.lda_precision;scores=x@p@m.T-.5*np.sum((m@p)*m,axis=1)
        else:scores=augment(self.features(x))@self.readout
        return np.asarray(self.classes)[scores.argmax(1)] if self.task=='classification' else scores

    @property
    def persistent_bytes(self):
        total=sum(p.numel()*p.element_size() for p in self.adapter.parameters())
        if self.head is not None:total+=sum(p.numel()*p.element_size() for p in self.head.parameters())
        if self.memory is not None:total+=self.memory.nbytes
        if self.buffer is not None:total+=sum(x.nbytes for x in self.buffer)+self.keys.nbytes
        for name in ('readout','rp','lda_means','lda_precision'):
            value=getattr(self,name,None)
            if value is not None:total+=value.nbytes
        return total

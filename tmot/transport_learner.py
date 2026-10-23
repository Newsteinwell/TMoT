"""WiMoT modules extended to growing labels and regression.

Complete-class domains use upstream domain experts. Sparse class/domain support
uses pooled observed class moments for a globally comparable OSCL readout.
No absent cell is invented. Calibration queries current TRAIN rows only.
"""
import copy
import numpy as np
import torch
from torch import nn
from .learner import supcon
from .moments import ClassificationMoments,RegressionMoments,affine_matrix,transport_risk,quadratic_risk,ridge_readout
from .wimot_core.statistics import TaskStatistics
from .wimot_core.readout import head_from,StudentDecoder,cases,case_key
from .wimot_core.calibration import folds,subtract,select


def transported(memory,classes,adapter):
    a=adapter.weight.detach().cpu().numpy().astype(float);b=adapter.bias.detach().cpu().numpy().astype(float)
    counts=[];means=[];covariances=[]
    for c in classes:
        cell=memory.cells[c]
        if cell.n<2:raise ValueError('Density readout requires >=2 observations in every observed class')
        mu=cell.total[:-1]/cell.n
        cov=(cell.gram[:-1,:-1]-cell.n*np.outer(mu,mu))/(cell.n-1)
        counts.append(cell.n);means.append(a@mu+b);covariances.append(a@((cov+cov.T)/2)@a.T)
    return TaskStatistics(np.asarray(counts,dtype=float),np.asarray(means),np.asarray(covariances))


class TransportLearner:
    def __init__(self,basis,epochs=50,readout='oscl',hist_weight=1.,sc_weight=.5,
                 ridge=.01,adapt=True,new_class_ce=True,seed=42):
        self.task,self.device,self.dimension=basis.task,basis.device,512
        self.epochs,self.readout_mode,self.hist_weight,self.sc_weight=epochs,readout,hist_weight,sc_weight
        self.ridge,self.adapt,self.new_class_ce=ridge,adapt,new_class_ce
        self.rng=np.random.default_rng(seed);self.seed=seed
        self.adapter=nn.Linear(512,128).to(self.device)
        with torch.no_grad():
            self.adapter.weight.zero_();self.adapter.weight[:,:256].copy_(basis.initial_adapter.weight)
            self.adapter.bias.copy_(basis.initial_adapter.bias)
        self.projector=copy.deepcopy(basis.initial_projector).to(self.device)
        self.head=copy.deepcopy(basis.initial_head).to(self.device)
        self.classes=list(basis.classes);self.domains=[];self.memory=None;self.stage=0;self.scale=None
        self.calibration=None;self.last_loss=None

    def _expand(self,n):
        if self.head.out_features==n:return
        old=self.head;self.head=nn.Linear(128,n).to(self.device)
        with torch.no_grad():
            self.head.weight[:old.out_features].copy_(old.weight);self.head.bias[:old.out_features].copy_(old.bias)

    def _f(self):
        f=torch.cat([self.head.weight@self.adapter.weight,(self.head.weight@self.adapter.bias+self.head.bias)[:,None]],1).double()
        return f-f.mean(0,keepdim=True) if self.task=='classification' else f

    def _risk(self,memory):
        g,k,v=memory.risk_moments(self.classes)
        if self.task=='classification':
            c=len(self.classes);center=(np.eye(c)-np.ones((c,c))/c)/np.sqrt(1-1/c)
            k=k@center;v=1.
        return [torch.as_tensor(t,dtype=torch.float64,device=self.device) for t in (g,k,v)]

    def observe(self,x,y):
        x=np.asarray(x,np.float32);y=np.asarray(y)
        torch.manual_seed(self.seed+self.stage*100)
        if self.task=='classification':
            new=sorted(set(y.tolist())-set(self.classes));self.classes.extend(new);self._expand(len(self.classes))
            local=np.array([self.classes.index(c) for c in y]);current=ClassificationMoments(512)
        else:
            new=[]
            if y.ndim==1:y=y[:,None]
            local=y.astype(np.float32);current=RegressionMoments(512,y.shape[1])
        current.update(x,y)
        if self.stage and self.adapt:
            tx=torch.as_tensor(x,device=self.device);ty=torch.as_tensor(local,device=self.device)
            constraints=[self._risk(m) for m in self.domains]
            parameters=list(self.adapter.parameters())+list(self.projector.parameters())+list(self.head.parameters())
            opt=torch.optim.AdamW(parameters,lr=.001,weight_decay=1e-4)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(self.epochs,1));losses=[]
            for epoch in range(self.epochs):
                order=self.rng.permutation(len(x))
                for start in range(0,len(x),128):
                    ids=order[start:start+128];h=self.adapter(tx[ids]);scores=self.head(h)
                    if self.task=='classification':
                        loss=self.sc_weight*supcon(self.projector(h),ty[ids],.07)
                        # Original DIL loss has no CE; genuinely new labels require explicit new-output supervision.
                        if new and self.new_class_ce:loss=loss+nn.functional.cross_entropy(scores,ty[ids])
                    else:loss=(scores-ty[ids]).square().sum(1).mean()
                    f=self._f()/self.scale
                    hist=torch.stack([quadratic_risk(f,*c) for c in constraints]).mean()
                    loss=loss+self.hist_weight*hist
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite adapted loss')
                    opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
                scheduler.step()
            self.last_loss=float(np.mean(losses[-int(np.ceil(len(x)/128)):])) if losses else None
        if self.scale is None:
            if self.task=='classification':
                f=self._f().detach();g=self._risk(current)[0]
                self.scale=float((f*(f@g)).sum().clamp_min(1e-16).sqrt())
            else:self.scale=1.
        self.domains.append(current)
        if self.memory is None:self.memory=copy.deepcopy(current)
        else:self.memory.update(x,y)
        g,k,_=self.memory.risk_moments(self.classes)
        matrix=affine_matrix(self.adapter.weight.detach().cpu().numpy(),self.adapter.bias.detach().cpu().numpy())
        g,k=transport_risk(g,k,matrix);self.linear_readout=ridge_readout(g,k,self.ridge)
        if self.task=='classification' and self.readout_mode=='oscl':self._fit_density(x,local)
        self.stage+=1

    def _fit_density(self,x,local):
        complete=all(set(m.cells)==set(self.classes) for m in self.domains)
        memories=self.domains if complete else [self.memory]
        stats=[transported(m,self.classes,self.adapter) for m in memories]
        task=len(stats)-1 if complete else 0
        h=self._h(x);tasks=np.full(len(x),task,dtype=int)
        self.calibration=dict(support='domain_experts' if complete else 'pooled_observed_classes',current_rows=len(x),historical_query_rows=0)
        # Two-fold train-only head calibration; tiny classes cannot estimate held-out covariance.
        counts=np.bincount(local,minlength=len(self.classes));positive=counts[counts>0]
        selected=(None,'gaussian')
        if positive.min()>=4:
            assign=folds(local,tasks,self.seed)
            probabilities={case_key(c):np.zeros((len(x),len(self.classes))) for c in cases()}
            for fold in (0,1):
                ids=np.flatnonzero(assign==fold)
                base=head_from(subtract(stats,h[ids],local[ids],tasks[ids]),len(self.classes))
                for case in cases():probabilities[case_key(case)][ids]=StudentDecoder(base,*case).predict_proba(h[ids])
            best,diagnostics=select(probabilities,local,tasks)
            selected=next(c for c in cases() if case_key(c)==best['case'])
            self.calibration.update(selected=best,diagnostics=diagnostics)
        else:self.calibration.update(selected={'case':case_key(selected)},fallback='fewer_than_four_current_rows_per_class')
        self.density=StudentDecoder(head_from(stats,len(self.classes)),*selected)

    def _h(self,x):
        with torch.no_grad():return self.adapter(torch.as_tensor(x,dtype=torch.float32,device=self.device)).cpu().numpy()

    def predict(self,x):
        h=self._h(x)
        if self.task=='classification' and self.readout_mode=='oscl':return np.asarray(self.classes)[self.density.predict_proba(h).argmax(1)]
        scores=np.column_stack([h,np.ones(len(h))])@self.linear_readout
        return np.asarray(self.classes)[scores.argmax(1)] if self.task=='classification' else scores

    @property
    def persistent_bytes(self):
        # Object graph accounting avoids double-counting shared decoder references.
        seen=set()
        def count(value):
            if id(value) in seen:return 0
            seen.add(id(value))
            if isinstance(value,np.ndarray):return value.nbytes
            if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
            if isinstance(value,dict):return sum(count(v) for v in value.values())
            if isinstance(value,(tuple,list)):return sum(count(v) for v in value)
            if hasattr(value,'__dict__'):return count(value.__dict__)
            return 0
        return count(self)

"""Arrival-only historical inputs for a matched support-coverage experiment."""
import hashlib
import numpy as np
from .local_moments import LocalMomentMixture
from .task_budget_replay import TaskBudgetER
from .gradient_diagnostic import empirical_sample

class DistillationSupport:
 def __init__(self,dimension,task,mode,seed=42):
  if mode not in ('current','local','er10'):raise ValueError(mode)
  self.task=task;self.mode=mode;self.stage=0;self.classes=[]
  self.local=LocalMomentMixture(dimension,task,seed+676,components=4,preserve_targets=True,support=4,min_cell=4) if mode=='local' else None
  self.real=TaskBudgetER(dimension,task,seed,mode='er') if mode=='er10' else None
 def sample(self,x,y,count=4096):
  rng=np.random.default_rng(1701+self.stage)
  history=False
  if self.stage and self.local is not None:
   hx,_=self.local.sample(count,self.classes,rng);history=True
  elif self.stage and self.real is not None and self.real.buffer is not None and len(self.real.buffer[0]):
   bx,by=self.real.buffer;labels=np.unique(by).tolist() if self.task=='classification' else []
   hx,_=empirical_sample(bx,by,count,rng,labels);history=True
  else:
   labels=np.unique(y).tolist() if self.task=='classification' else []
   hx,_=empirical_sample(x,y,count,rng,labels)
  hx=np.asarray(hx,np.float32)
  return hx,dict(historical_support=history,support_count=len(hx),support_sha256=hashlib.sha256(hx.tobytes()).hexdigest())
 def update(self,x,y):
  if self.local is not None:self.local.update(x,y)
  if self.real is not None:self.real.store_buffer(x,y);self.real.stage+=1
  if self.task=='classification':self.classes.extend(sorted(set(y.tolist())-set(self.classes)))
  self.stage+=1
 @property
 def nbytes(self):
  if self.local is not None:return self.local.nbytes
  if self.real is not None:return sum(v.nbytes for v in self.real.buffer)+self.real.keys.nbytes if self.real.buffer is not None else 0
  return 0
 @property
 def audit(self):return self.real.buffer_audit[-1] if self.real is not None else None

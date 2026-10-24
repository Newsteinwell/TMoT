"""Strict task-wise real-example quotas, including zero-capacity arrivals."""
import numpy as np
from .replay_functional import ReplayFunctionalLearner,representative_weights
from .regularized_innovation import RegularizedInnovationLearner
from .gradient_diagnostic import empirical_sample

class TaskBudgetMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.selection_rng=np.random.default_rng(self.seed+90217)
        self.buffer_audit=[]
    def sample_buffer(self):
        if self.buffer is None or not len(self.buffer[0]):return None
        classes=np.unique(self.buffer[1]).tolist() if self.task=='classification' else []
        return empirical_sample(*self.buffer,self.replay_count,self.rng,classes)
    def replay_weights(self,y,hy):
        # Condition risk on represented classes. Never invent examples for absent classes.
        if hy is None:
            return representative_weights(y,None,self.task,{},0)
        if self.task=='regression':return representative_weights(y,hy,self.task,self.counts,self.seen)
        available=set(np.unique(hy).tolist())
        counts={c:n for c,n in self.counts.items() if c in available}
        return representative_weights(y,hy,self.task,counts,self.seen)
    def store_buffer(self,x,y):
        quota=len(x)//10
        if self.task=='classification':
            pools=[self.selection_rng.permutation(np.flatnonzero(y==c)).tolist() for c in np.unique(y)]
            chosen=[]
            while len(chosen)<quota:
                for k in self.selection_rng.permutation(len(pools)):
                    if pools[k]:chosen.append(pools[k].pop())
                    if len(chosen)==quota:break
            ids=np.asarray(chosen,dtype=int)
        else:ids=self.selection_rng.choice(len(x),size=quota,replace=False)
        before=0 if self.buffer is None else len(self.buffer[0])
        bx=x[ids].copy();by=y[ids].copy()
        self.buffer=(bx,by) if self.buffer is None else (np.concatenate([self.buffer[0],bx]),np.concatenate([self.buffer[1],by]))
        self.keys=np.zeros(len(self.buffer[0]),dtype=np.float64)
        self.buffer_audit.append(dict(stage=self.stage+1,current_samples=len(x),quota=quota,selected_indices=ids.tolist(),before=before,after=len(self.buffer[0])))

class TaskBudgetER(TaskBudgetMixin,ReplayFunctionalLearner):
    pass

class TaskBudgetMomentER(TaskBudgetMixin,RegularizedInnovationLearner):
    pass

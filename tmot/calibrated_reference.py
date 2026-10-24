"""First-task-only bounded persistence coefficient; no validation selection."""
import numpy as np
from .causal_reference import CausalReference

class CalibratedReference(CausalReference):
    def __init__(self):
        super().__init__('append')
        self.mode='calibrated'
    def fit(self,past,target):
        p=np.asarray(past,np.float64);y=np.asarray(target,np.float64)
        if p.ndim!=2 or p.shape!=y.shape or len(p)<1 or not np.isfinite(p).all() or not np.isfinite(y).all():
            raise ValueError('Expected finite aligned observations and targets')
        if hasattr(self,'mean'):raise RuntimeError('First-arrival calibration is immutable')
        # Keep input scaling identical to CausalReference, including input dtype.
        super().fit(past)
        x=p-p.mean(0);z=y-y.mean(0);den=np.sum(x*x,axis=0);num=np.sum(x*z,axis=0)
        self.unclipped_beta=np.divide(num,den,out=np.zeros_like(num),where=den>0)
        self.beta=np.clip(self.unclipped_beta,0.,1.)
        self.calibration_count=len(p)
        return self
    def offset(self,past):return np.asarray(past)*self.beta

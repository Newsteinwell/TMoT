"""Factorial covariance approximation with matched primary sampling streams."""
from dataclasses import dataclass
import numpy as np
from .local_moments import LocalCell,LocalMomentMixture
from .regularized_innovation import RegularizedInnovationLearner
VARIANTS=('rank16_shrink01','rank16_shrink0','full_shrink01','full_shrink0')

@dataclass
class CovarianceCell(LocalCell):
    reference_width: int=0
    extra_rng: object=None
    def sample(self,n,rng):
        # Consume exactly the legacy draws even when the new factor is wider.
        low=rng.normal(size=(n,self.reference_width));diagonal=rng.normal(size=(n,len(self.mean)))
        width=self.factor.shape[1]
        if width>self.reference_width:low=np.c_[low,self.extra_rng.normal(size=(n,width-self.reference_width))]
        else:low=low[:,:width]
        return (self.mean+low@self.factor.T+diagonal*self.residual_sd).astype(np.float32)

class CovarianceMixture(LocalMomentMixture):
    def __init__(self,*args,covariance='rank16_shrink01',**kwargs):
        if covariance not in VARIANTS:raise ValueError(covariance)
        super().__init__(*args,**kwargs);self.covariance=covariance;self.cell_serial=0
    def fit_cell(self,batch):
        reference=super().fit_cell(batch)
        full=self.covariance.startswith('full');shrinkage=.1 if self.covariance.endswith('shrink01') else 0.
        rank=max(len(batch)-1,0) if full else self.rank
        if self.covariance=='rank16_shrink01':cell=reference
        elif self.preserve_targets and self.task=='regression':cell=LocalCell.fit_target_preserving(batch,self.dimension,rank,shrinkage)
        else:cell=LocalCell.fit(batch,rank,shrinkage)
        if full and shrinkage==0:
            # Full factors already span the empirical centered covariance;
            # do not turn subtraction roundoff into artificial diagonal noise.
            cell.residual_sd=np.zeros_like(cell.residual_sd)
        extra=np.random.default_rng(104729+self.cell_serial*7919);self.cell_serial+=1
        return CovarianceCell(cell.n,cell.mean,cell.factor,cell.residual_sd,reference.factor.shape[1],extra)

class CovarianceReplayLearner(RegularizedInnovationLearner):
    def __init__(self,*args,covariance='rank16_shrink01',**kwargs):
        super().__init__(*args,**kwargs)
        if self.mode!='local':raise ValueError('Covariance ablation uses local replay')
        self.covariance=covariance
        self.local=CovarianceMixture(self.dimension,self.task,self.seed+676,components=4,preserve_targets=True,support=4,min_cell=4,covariance=covariance)

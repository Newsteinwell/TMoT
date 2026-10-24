"""Unbounded-history diagnostic with identical local-cell allocations."""
from dataclasses import dataclass
import numpy as np
from .covariance_replay import CovarianceCell,CovarianceMixture
from .regularized_innovation import RegularizedInnovationLearner
VARIANTS=('gaussian','empirical')
@dataclass
class EmpiricalCell(CovarianceCell):
    observed: object=None
    sample_rng: object=None
    empirical: bool=False
    def sample(self,n,rng):
        gaussian=super().sample(n,rng)
        return self.observed[self.sample_rng.integers(0,len(self.observed),n)].copy() if self.empirical else gaussian
    @property
    def nbytes(self):return super().nbytes+self.observed.nbytes
class EmpiricalMixture(CovarianceMixture):
    def __init__(self,*args,variant='gaussian',**kwargs):
        if variant not in VARIANTS:raise ValueError(variant)
        self.variant=variant;super().__init__(*args,**kwargs)
    def fit_cell(self,batch):
        cell=super().fit_cell(batch)
        return EmpiricalCell(cell.n,cell.mean,cell.factor,cell.residual_sd,cell.reference_width,cell.extra_rng,np.asarray(batch,np.float32).copy(),np.random.default_rng(15485863+7919*self.cell_serial),self.variant=='empirical')
class EmpiricalReplayLearner(RegularizedInnovationLearner):
    def __init__(self,*args,variant='gaussian',**kwargs):
        super().__init__(*args,**kwargs);assert self.mode=='local';self.variant=variant
        self.local=EmpiricalMixture(self.dimension,self.task,self.seed+676,components=4,preserve_targets=True,support=4,min_cell=4,variant=variant)

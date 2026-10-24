"""Change latent higher moments without changing stored mixture moments."""
from dataclasses import dataclass
import numpy as np
from .local_moments import LocalCell,LocalMomentMixture
from .regularized_innovation import RegularizedInnovationLearner

@dataclass
class LatentCell(LocalCell):
    law: str='gaussian'
    def sample(self,n,rng):
        low=rng.normal(size=(n,self.factor.shape[1]));diagonal=rng.normal(size=(n,len(self.mean)))
        if self.law=='rademacher':
            low=np.where(low<0,-1.,1.);diagonal=np.where(diagonal<0,-1.,1.)
        elif self.law!='gaussian':raise ValueError(self.law)
        return (self.mean+low@self.factor.T+diagonal*self.residual_sd).astype(np.float32)

class LatentMomentMixture(LocalMomentMixture):
    def __init__(self,*args,law='gaussian',**kwargs):
        if law not in ('gaussian','rademacher'):raise ValueError(law)
        super().__init__(*args,**kwargs);self.law=law
    def update(self,x,y):
        super().update(x,y)
        for label,cells in self.cells.items():
            self.cells[label]=[c if isinstance(c,LatentCell) else LatentCell(c.n,c.mean,c.factor,c.residual_sd,self.law) for c in cells]

class LatentReplayLearner(RegularizedInnovationLearner):
    def __init__(self,*args,latent_law='gaussian',**kwargs):
        super().__init__(*args,**kwargs);self.latent_law=latent_law
        if self.mode=='local':
            self.local=LatentMomentMixture(self.dimension,self.task,self.seed+676,components=4,preserve_targets=True,support=4,min_cell=4,law=latent_law)
        elif latent_law!='gaussian':raise ValueError('ER has no synthetic latent law')

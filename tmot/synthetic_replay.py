"""Per-cell synthetic joint representatives fitted only to arriving data.

Component count still grows with arrivals; no fixed-total-memory claim.
"""
from dataclasses import dataclass
import numpy as np
import torch
from .covariance_replay import CovarianceCell,CovarianceMixture
from .regularized_innovation import RegularizedInnovationLearner
VARIANTS=('gaussian','finite','matched')
def fit_points(rows,initial,input_dim,seed,steps=80):
    rows=np.asarray(rows,np.float32);mean=rows.mean(0);sd=rows.std(0);sd=np.maximum(sd,max(float(np.median(sd))*.1,1e-4))
    rng=np.random.default_rng(seed);projection=torch.tensor((rng.normal(size=(input_dim,64))/np.sqrt(input_dim)).astype(np.float32));bias=torch.tensor(rng.normal(size=64).astype(np.float32))
    real=torch.tensor((rows-mean)/sd);points=torch.tensor((initial-mean)/sd,requires_grad=True)
    def features(z):return torch.cat([z[:,:input_dim]@projection,torch.relu(z[:,:input_dim]@projection+bias),z[:,input_dim:]],1)
    with torch.no_grad():
        f=features(real);fm=f.mean(0);fg=f.T@f/len(f);rm=real.mean(0);rv=(real**2).mean(0)
        xy=real.T@real[:,input_dim:]/len(real)
    def loss(z):
        f=features(z);value=(f.mean(0)-fm).square().mean()+(f.T@f/len(f)-fg).square().mean()+.1*((z.mean(0)-rm).square().mean()+((z**2).mean(0)-rv).square().mean())
        if z.shape[1]>input_dim:value=value+(z.T@z[:,input_dim:]/len(z)-xy).square().mean()
        return value
    initial_loss=float(loss(points).detach());best=points.detach().clone();best_loss=initial_loss;opt=torch.optim.Adam([points],lr=.05)
    for _ in range(steps):
        opt.zero_grad();value=loss(points);value.backward();opt.step()
        with torch.no_grad():
            score=float(loss(points))
            if np.isfinite(score) and score<best_loss:best_loss=score;best=points.detach().clone()
    return (best.numpy()*sd+mean).astype(np.float32),dict(initial=initial_loss,final=best_loss,steps=steps)
@dataclass
class SyntheticCell(CovarianceCell):
    points: object=None
    point_rng: object=None
    variant: str='gaussian'
    fitting: object=None
    def sample(self,n,rng):
        generated=super().sample(n,rng)
        return generated if self.variant=='gaussian' else self.points[self.point_rng.integers(0,len(self.points),n)].copy()
    @property
    def nbytes(self):return super().nbytes+(0 if self.points is None else self.points.nbytes)
class SyntheticMixture(CovarianceMixture):
    def __init__(self,*args,variant='gaussian',**kwargs):
        if variant not in VARIANTS:raise ValueError(variant)
        self.variant=variant;super().__init__(*args,**kwargs)
    def fit_cell(self,batch):
        cell=super().fit_cell(batch);seed=32452843+self.cell_serial*7919;points=None;fitting=None
        if self.variant!='gaussian':
            points=cell.sample(16,np.random.default_rng(seed))
            if self.variant=='matched':points,fitting=fit_points(batch,points,self.dimension,seed+1)
        return SyntheticCell(cell.n,cell.mean,cell.factor,cell.residual_sd,cell.reference_width,cell.extra_rng,points,np.random.default_rng(seed+2),self.variant,fitting)
class SyntheticReplayLearner(RegularizedInnovationLearner):
    def __init__(self,*args,variant='gaussian',**kwargs):
        super().__init__(*args,**kwargs);assert self.mode=='local';self.variant=variant
        self.local=SyntheticMixture(self.dimension,self.task,self.seed+676,components=4,preserve_targets=True,support=4,min_cell=4,variant=variant)

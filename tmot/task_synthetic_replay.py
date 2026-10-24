"""Synthetic representatives matching current learned features, arrival-only."""
import numpy as np
import torch
from .synthetic_replay import SyntheticCell,fit_points
from .covariance_replay import CovarianceMixture
from .regularized_innovation import RegularizedInnovationLearner
VARIANTS=('gaussian','finite','random','task')
def fit_task_points(rows,initial,input_dim,teacher,steps=80):
    rows=np.asarray(rows,np.float32);mean=rows.mean(0);sd=rows.std(0);sd=np.maximum(sd,max(float(np.median(sd))*.1,1e-4))
    real=torch.tensor((rows-mean)/sd);points=torch.tensor((initial-mean)/sd,requires_grad=True)
    mu=torch.tensor(mean);scale=torch.tensor(sd);input_mean=torch.tensor(teacher.mean);input_sd=torch.tensor(teacher.sd);hidden_mean=torch.tensor(teacher.hidden_mean)
    assert all(not p.requires_grad for p in teacher.net.parameters())
    def features(z):
        raw=z[:,:input_dim]*scale[:input_dim]+mu[:input_dim]
        h=(teacher.net((raw-input_mean)/input_sd)-hidden_mean)/teacher.hidden_scale
        return torch.cat([h,z[:,input_dim:]],1)
    with torch.no_grad():
        f=features(real);fm=f.mean(0);fg=f.T@f/len(f);rm=real.mean(0);rv=(real**2).mean(0);xy=real.T@real[:,input_dim:]/len(real)
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
class TaskSyntheticMixture(CovarianceMixture):
    def __init__(self,*args,variant='gaussian',points_per_cell=16,**kwargs):
        if variant not in VARIANTS:raise ValueError(variant)
        assert isinstance(points_per_cell,int) and points_per_cell>0
        self.points_per_cell=points_per_cell;self.variant=variant;self.teacher=None;super().__init__(*args,**kwargs)
    def fit_cell(self,batch):
        cell=super().fit_cell(batch);seed=32452843+self.cell_serial*7919;points=None;fitting=None
        if self.variant!='gaussian':
            points=cell.sample(self.points_per_cell,np.random.default_rng(seed))
            if self.variant=='random':points,fitting=fit_points(batch,points,self.dimension,seed+1)
            elif self.variant=='task':
                assert self.teacher is not None
                points,fitting=fit_task_points(batch,points,self.dimension,self.teacher)
        return SyntheticCell(cell.n,cell.mean,cell.factor,cell.residual_sd,cell.reference_width,cell.extra_rng,points,np.random.default_rng(seed+2),self.variant,fitting)
class TaskSyntheticLearner(RegularizedInnovationLearner):
    def __init__(self,*args,variant='gaussian',points_per_cell=16,**kwargs):
        super().__init__(*args,**kwargs);assert self.mode=='local';self.variant=variant;self.points_per_cell=points_per_cell
        self.local=TaskSyntheticMixture(self.dimension,self.task,self.seed+676,components=4,preserve_targets=True,support=4,min_cell=4,variant=variant,points_per_cell=points_per_cell)
    def update_local(self,x,y):
        self.local.teacher=self.function
        try:self.local.update(x,y)
        finally:self.local.teacher=None

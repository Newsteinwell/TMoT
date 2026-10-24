"""Read-only prefix-TRAIN diagnostics; historical rows live outside the learner."""
import numpy as np
import torch
from .local_moments import local_partition,merge_small_cells
from .covariance_replay import CovarianceCell
class DriftDiagnostic:
    def __init__(self,dimension,task,seed=42):
        self.dimension=dimension;self.task=task;self.rng=np.random.default_rng(seed+676);self.offset={};self.cells=[];self.rows=[]
    def add(self,x,y,learner,stage):
        x=np.asarray(x);y=np.asarray(y);cl=self.task=='classification'
        for label in (np.unique(y) if cl else [None]):
            mask=y==label if cl else np.ones(len(x),bool);inputs=x[mask];ys=y[mask];ys=ys[:,None] if ys.ndim==1 else ys
            raw=inputs if cl else np.c_[inputs,ys];assignment=merge_small_cells(inputs,local_partition(inputs,min(4,max(1,len(inputs)//4)),self.rng),4)
            for component in np.unique(assignment):
                batch=raw[assignment==component];idx=self.offset.get(label,0);cell=learner.local.cells[label][idx];self.offset[label]=idx+1;serial=len(self.cells)+1
                assert cell.n==len(batch);np.testing.assert_array_equal(batch.astype(float).mean(0).astype(np.float32),cell.mean)
                initial=CovarianceCell.sample(cell,learner.points_per_cell,np.random.default_rng(32452843+serial*7919))
                self.cells.append(dict(cell=cell,real=batch.astype(np.float32),initial=initial,label=None if label is None else int(label),birth=stage,serial=serial))
        assert self.rng.bit_generator.state==learner.local.rng.bit_generator.state
    def evaluate(self,learner,stage):
        teacher=learner.function;dim=self.dimension;result=[]
        with torch.no_grad():
            for item in self.cells:
                raw=item['real'];cell=item['cell'];mean=raw.mean(0);sd=raw.std(0);sd=np.maximum(sd,max(float(np.median(sd))*.1,1e-4));mu=torch.tensor(mean);scale=torch.tensor(sd)
                real=torch.tensor((raw-mean)/sd);im=torch.tensor(teacher.mean);isd=torch.tensor(teacher.sd);hm=torch.tensor(teacher.hidden_mean)
                def feature(z):
                    x=z[:,:dim]*scale[:dim]+mu[:dim];h=(teacher.net((x-im)/isd)-hm)/teacher.hidden_scale
                    return torch.cat([h,z[:,dim:]],1)
                f=feature(real);fm=f.mean(0);fg=f.T@f/len(f);rm=real.mean(0);rv=(real**2).mean(0);xy=real.T@real[:,dim:]/len(real)
                def measure(points):
                    z=torch.tensor((points-mean)/sd);ff=feature(z);mean_error=(ff.mean(0)-fm).square().mean();gram_error=(ff.T@ff/len(ff)-fg).square().mean()
                    value=mean_error+gram_error+.1*((z.mean(0)-rm).square().mean()+((z**2).mean(0)-rv).square().mean())
                    if z.shape[1]>dim:value=value+(z.T@z[:,dim:]/len(z)-xy).square().mean()
                    return float(value),float(torch.linalg.norm(ff.mean(0)-fm)/torch.linalg.norm(fm).clamp_min(1e-12)),float(torch.linalg.norm(ff.T@ff/len(ff)-fg)/torch.linalg.norm(fg).clamp_min(1e-12))
                initial,mi,gi=measure(item['initial']);optimized,mo,go=measure(cell.points)
                if item['birth']==stage:
                    np.testing.assert_allclose(initial,cell.fitting['initial'],rtol=1e-4,atol=1e-6);np.testing.assert_allclose(optimized,cell.fitting['final'],rtol=1e-4,atol=1e-6)
                weight=cell.n/(len(learner.classes)*learner.counts[item['label']]) if self.task=='classification' else cell.n/learner.seen
                result.append(dict(stage=stage,cell=item['serial'],birth=item['birth'],age=stage-item['birth'],n=cell.n,label=item['label'],weight=weight,initial_loss=initial,optimized_loss=optimized,initial_mean_relative=mi,optimized_mean_relative=mo,initial_gram_relative=gi,optimized_gram_relative=go))
        np.testing.assert_allclose(sum(r['weight'] for r in result),1,atol=1e-12);self.rows.extend(result);return result

"""Offline gradient comparison at fixed final checkpoints; not a training method."""
import numpy as np
import torch
from .moments import augment
from .nonlinear_moments import per_sample_loss

def risk_gradient(model,x,y,batch=256):
    x=np.asarray(x,np.float32);y=np.asarray(y);model.net.zero_grad(set_to_none=True)
    if model.task=='classification':
        ids=np.array([model.classes.index(c) for c in y]);counts=np.bincount(ids,minlength=len(model.classes))
        if np.any(counts==0):raise ValueError('gradient comparison requires all seen classes')
        targets=np.eye(len(model.classes),dtype=np.float32)[ids]
        weights=1./(len(model.classes)*counts[ids])
    else:
        targets=y.astype(np.float32);weights=np.full(len(x),1./len(x))
    loss_value=0.
    for start in range(0,len(x),batch):
        sl=slice(start,start+batch)
        z=torch.as_tensor((x[sl]-model.mean)/model.sd,device=model.device)
        baseline=torch.as_tensor(augment(x[sl])@model.base,dtype=torch.float32,device=model.device)
        delta=model.net(z);pred=baseline+model.alpha*delta
        target=torch.as_tensor(targets[sl],device=model.device)
        losses=per_sample_loss(pred,target,model.task,model.loss)+model.residual_weight*delta.square().mean(1)
        loss=(losses*torch.as_tensor(weights[sl],dtype=torch.float32,device=model.device)).sum()
        loss.backward();loss_value+=float(loss.detach())
    grad=torch.cat([p.grad.detach().flatten() for p in model.net.parameters()]).cpu().numpy().astype(np.float64)
    model.net.zero_grad(set_to_none=True)
    return loss_value,grad

def compare_gradients(reference,estimate):
    nr=np.linalg.norm(reference);ne=np.linalg.norm(estimate);dot=float(reference@estimate)
    return dict(reference_norm=float(nr),estimate_norm=float(ne),cosine=float(dot/(nr*ne)) if nr*ne>1e-20 else None,
                relative_error=float(np.linalg.norm(estimate-reference)/nr) if nr>1e-10 else None,
                negative_step_directional_derivative=-dot,reference_near_zero=bool(nr<1e-8))

def empirical_sample(x,y,count,rng,classes):
    if classes:
        sizes=rng.multinomial(count,np.ones(len(classes))/len(classes))
        ids=np.concatenate([rng.choice(np.flatnonzero(y==c),size=n,replace=True) for c,n in zip(classes,sizes)])
    else:ids=rng.integers(len(x),size=count)
    return x[ids],y[ids]

class CachedGaussian:
    def __init__(self,cell):
        self.mean=cell.total[:-1]/cell.n
        covariance=cell.gram[:-1,:-1]/cell.n-np.outer(self.mean,self.mean)
        covariance=(covariance+covariance.T)/2
        covariance=.9*covariance+.1*np.diag(np.maximum(np.diag(covariance),0))
        values,vectors=np.linalg.eigh(covariance)
        self.factor=vectors*np.sqrt(np.maximum(values,0))
    def sample(self,count,rng):
        return (self.mean+rng.normal(size=(count,len(self.mean)))@self.factor.T).astype(np.float32)

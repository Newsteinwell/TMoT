"""Empirical-Bayes linear readout from cumulative moments, without raw replay.

Gaussian evidence is exact for the weighted linear observation model; one-hot
classification is a surrogate, not a calibrated categorical likelihood.
"""
import numpy as np
import torch
from .moments import ClassificationMoments, RegressionMoments


def evidence_objective(scatter, cross, target_scatter, count, log_alpha, log_beta):
    d, outputs = cross.shape
    alpha = log_alpha.exp().expand(d)
    beta = log_beta.exp()
    precision = torch.diag(alpha) + beta*scatter
    factor = torch.linalg.cholesky(precision)
    mean = torch.cholesky_solve(beta*cross, factor)
    logdet = 2*factor.diag().log().sum()
    loss = .5*(outputs*(logdet-alpha.log().sum()-count*log_beta)
               + beta*target_scatter-beta*(cross*mean).sum())
    return loss, mean


class EvidenceReadout:
    def __init__(self, dimension, task='classification', mode='isotropic', steps=60, device='cuda', refine=False):
        if mode not in ('isotropic','ard'): raise ValueError(mode)
        self.dimension,self.task,self.mode,self.steps,self.device=dimension,task,mode,steps,device
        self.refine=refine
        self.classes=[];self.memory=None;self.last_loss=None;self.diagnostics=None

    def observe(self, x, y):
        y=np.asarray(y)
        if self.task=='classification':
            self.classes.extend(sorted(set(y.tolist())-set(self.classes)))
            if self.memory is None:self.memory=ClassificationMoments(self.dimension)
        else:
            if y.ndim==1:y=y[:,None]
            if self.memory is None:self.memory=RegressionMoments(self.dimension,y.shape[1])
        self.memory.update(x,y)
        g,k,v=self.memory.risk_moments(self.classes)
        n=sum(c.n for c in self.memory.cells.values()) if self.task=='classification' else self.memory.n
        self.x_mean=g[:-1,-1].copy();self.y_mean=k[-1].copy()
        scatter=n*(g[:-1,:-1]-np.outer(self.x_mean,self.x_mean))
        scatter=(scatter+scatter.T)/2
        cross=n*(k[:-1]-np.outer(self.x_mean,self.y_mean))
        target_scatter=max(float(n*(v-self.y_mean@self.y_mean)),0.)
        if n<2:raise ValueError('Evidence readout requires at least two observations')
        tensors=[torch.as_tensor(t,dtype=torch.float64,device=self.device) for t in (scatter,cross,target_scatter)]
        # Refit the same deterministic objective each arrival; no validation labels.
        la=torch.zeros(1 if self.mode=='isotropic' else self.dimension,dtype=torch.float64,device=self.device,requires_grad=True)
        lb=torch.tensor(float(np.clip(np.log(1/max(target_scatter/((n-1)*cross.shape[1]),1e-6)),-12,12)),dtype=torch.float64,device=self.device,requires_grad=True)
        optimizer=torch.optim.Adam([la,lb],lr=.1)
        best=None;initial=None;history=[]
        for step in range(self.steps+1):
            loss,mean=evidence_objective(*tensors,n-1,la,lb)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite evidence')
            value=float(loss.detach());history.append(value)
            if initial is None:initial=value
            if best is None or value<best[0]:best=(value,mean.detach().cpu().numpy().copy(),la.detach().cpu().numpy().copy(),float(lb.detach()))
            if step==self.steps:break
            optimizer.zero_grad();loss.backward();optimizer.step()
            with torch.no_grad():la.clamp_(-12,12);lb.clamp_(-12,12)
        refinement={}
        if self.refine:
            from scipy.optimize import minimize
            normalizer=(n-1)*cross.shape[1]
            adam_best=best[0]
            def evaluate(theta):
                nonlocal best
                a=torch.tensor(theta[:-1],dtype=torch.float64,device=self.device,requires_grad=True)
                b=torch.tensor(theta[-1],dtype=torch.float64,device=self.device,requires_grad=True)
                value,posterior=evidence_objective(*tensors,n-1,a,b)
                if not torch.isfinite(value):raise FloatingPointError('Nonfinite refinement evidence')
                raw=float(value.detach());value.backward()
                gradient=np.r_[a.grad.detach().cpu().numpy(),float(b.grad.detach())]/normalizer
                if raw<best[0]:best=(raw,posterior.detach().cpu().numpy().copy(),theta[:-1].copy(),float(theta[-1]))
                return raw/normalizer,gradient
            start=np.r_[best[2],best[3]]
            initial_gradient=evaluate(start)[1]
            result=minimize(evaluate,start,method='L-BFGS-B',jac=True,bounds=[(-12.,12.)]*len(start),
                            options=dict(maxiter=300,maxls=30,gtol=1e-6,ftol=1e-12))
            theta=np.r_[best[2],best[3]];gradient=evaluate(theta)[1]
            projected=gradient.copy()
            projected[((theta<=-12+1e-8)&(gradient>0))|((theta>=12-1e-8)&(gradient<0))]=0
            refinement=dict(adam_best_evidence=adam_best,refined_evidence=best[0],
                            optimizer_success=bool(result.success),optimizer_message=str(result.message),
                            optimizer_iterations=int(result.nit),optimizer_evaluations=int(result.nfev),
                            initial_normalized_max_gradient=float(np.abs(initial_gradient).max()),
                            normalized_projected_gradient=float(np.abs(projected).max()),
                            gradient_tolerance_met=bool(np.abs(projected).max()<=1e-6),
                            active_bounds=int(np.sum((theta<=-12+1e-8)|(theta>=12-1e-8))))
        self.last_loss,self.readout,self.log_alpha,self.log_beta=best
        self.diagnostics=dict(refinement=refinement,initial_evidence=initial,best_evidence=best[0],last_evidence=history[-1],steps=self.steps,
                              observations=n,weighting='class-balanced weighted Gaussian surrogate' if self.task=='classification' else 'sample-weighted Gaussian',
                              alpha_min=float(np.exp(self.log_alpha).min()),alpha_max=float(np.exp(self.log_alpha).max()),beta=float(np.exp(self.log_beta)))

    def predict(self, x):
        score=(np.asarray(x)-self.x_mean)@self.readout+self.y_mean
        return np.asarray(self.classes)[score.argmax(1)] if self.task=='classification' else score

    @property
    def persistent_bytes(self):
        return self.memory.nbytes+sum(x.nbytes for x in (self.x_mean,self.y_mean,self.readout,self.log_alpha))+8

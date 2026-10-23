"""Direct local statistical experts: categorical densities or conditional means.

Gaussian conditional regression is an established GMR building block, not an
originality claim. Student marginal gates use WiMoT's equal-covariance formula.
No pseudo-samples, historical rows, task ID at inference, or iterative readout.
"""
import numpy as np
from scipy.special import gammaln,logsumexp
from .local_moments import LocalMomentMixture


class LocalExpert:
    def __init__(self,cell,dimension,ridge):
        self.mean=cell.mean[:dimension]
        factor=cell.factor[:dimension].astype(float)
        variance=np.sum(factor**2,axis=1)+cell.residual_sd[:dimension].astype(float)**2
        scale=max(float(variance.mean()),1e-8)
        noise=cell.residual_sd[:dimension].astype(float)**2+max(ridge,1e-8)*scale
        self.invdiag=1/noise
        self.weighted=factor*self.invdiag[:,None]
        middle=np.eye(factor.shape[1])+factor.T@self.weighted
        self.middle_inv=np.linalg.inv(middle)
        self.logdet=float(np.log(noise).sum()+np.linalg.slogdet(middle)[1])
        self.target_mean=cell.mean[dimension:]
        cross=factor@cell.factor[dimension:].astype(float).T
        self.beta=self.solve(cross)
        self.n=cell.n

    def solve(self,rhs):
        return self.invdiag[:,None]*rhs-self.weighted@(self.middle_inv@(self.weighted.T@rhs))

    def log_density(self,x,df=None):
        delta=np.asarray(x,dtype=float)-self.mean
        projected=delta@self.weighted
        quadratic=np.maximum(np.sum(delta**2*self.invdiag,axis=1)-np.sum((projected@self.middle_inv)*projected,axis=1),0)
        d=len(self.mean)
        if df is None:return -.5*(quadratic+self.logdet+d*np.log(2*np.pi))
        if df<=2:raise ValueError('Student covariance requires df>2')
        scale=(df-2)/df
        return (gammaln((df+d)/2)-gammaln(df/2)-.5*(d*np.log(df*np.pi)+self.logdet+d*np.log(scale))
                -.5*(df+d)*np.log1p(quadratic/(df*scale)))

    def conditional_mean(self,x):return self.target_mean+(np.asarray(x,dtype=float)-self.mean)@self.beta

    @property
    def cache_bytes(self):return self.invdiag.nbytes+self.weighted.nbytes+self.middle_inv.nbytes+self.beta.nbytes+16


class GatedLocalLearner:
    def __init__(self,dimension,task='classification',seed=42,ridge=.1,components=4,density='student'):
        if density not in ('gaussian','student'):raise ValueError(density)
        self.dimension,self.task,self.ridge=dimension,task,ridge
        self.df=10. if density=='student' else None
        self.local=LocalMomentMixture(dimension,task,seed+676,components=components,preserve_targets=True)
        self.classes=[];self.last_loss=None;self.experts=[]

    def observe(self,x,y):
        if self.task=='classification':self.classes.extend(sorted(set(np.asarray(y).tolist())-set(self.classes)))
        self.local.update(x,y);self.experts=[];priors=[];labels=[]
        for label,cells in self.local.cells.items():
            total=sum(c.n for c in cells)
            for cell in cells:
                self.experts.append(LocalExpert(cell,self.dimension,self.ridge));labels.append(label)
                priors.append(cell.n/total/len(self.classes) if self.task=='classification' else cell.n)
        priors=np.array(priors,dtype=float);priors/=priors.sum()
        self.log_priors=np.log(priors);self.labels=labels

    def probabilities(self,x):
        logits=np.column_stack([e.log_density(x,self.df) for e in self.experts])+self.log_priors
        return np.exp(logits-logsumexp(logits,axis=1,keepdims=True))

    def predict(self,x):
        weights=self.probabilities(x)
        if self.task=='classification':
            scores=np.zeros((len(x),len(self.classes)))
            for j,c in enumerate(self.labels):scores[:,self.classes.index(c)]+=weights[:,j]
            return np.asarray(self.classes)[scores.argmax(1)]
        return np.einsum('nk,nkp->np',weights,np.stack([e.conditional_mean(x) for e in self.experts],axis=1))

    @property
    def persistent_bytes(self):return self.local.nbytes+sum(e.cache_bytes for e in self.experts)+self.log_priors.nbytes

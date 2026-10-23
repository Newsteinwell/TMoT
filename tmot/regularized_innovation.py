"""Regularized residual features with correction fitted to the base residual."""
import numpy as np
from .exact_base_completion import ExactBaseFunctionalLearner,spectral
from .moments import ridge_readout


def regularized_innovation_readout(g,k,dimension,ridge):
    if ridge<=0:raise ValueError('Positive regularization is required')
    phi=np.r_[np.arange(dimension),len(g)-1];hi=np.arange(dimension,len(g)-1)
    a=g[np.ix_(phi,phi)];u=g[np.ix_(phi,hi)]
    base=ridge_readout(a,k[phi],ridge)
    b=ridge_readout(a,u,ridge)
    overlap=u-a@b
    raw=g[np.ix_(hi,hi)]-b.T@u-u.T@b+b.T@a@b
    residual=spectral(raw)[0]
    # Residual features are not exactly orthogonal under regularized projection.
    # Fit the residual of the fixed base prediction, not the full target again.
    cross=k[hi]-b.T@k[phi]-overlap.T@base
    correction=np.linalg.solve(residual+ridge*np.eye(len(hi)),cross)
    readout=np.zeros_like(k);readout[phi]=base-b@correction;readout[hi]=correction
    return readout,dict(projection_norm=float(np.linalg.norm(b)),
        overlap_norm=float(np.linalg.norm(overlap)),residual_psd_adjustment=float(np.max(np.abs(raw-residual))),
        base_weight_norm=float(np.linalg.norm(base)),correction_weight_norm=float(np.linalg.norm(correction)))


class RegularizedInnovationLearner(ExactBaseFunctionalLearner):
    def __init__(self,*args,regularized=True,**kwargs):
        super().__init__(*args,completion=True,**kwargs);self.regularized=regularized;self.innovation_diagnostics=None
    def observe(self,x,y):
        super().observe(x,y)
        if self.regularized:self.readout,self.innovation_diagnostics=regularized_innovation_readout(self.g,self.k,self.dimension,self.ridge)

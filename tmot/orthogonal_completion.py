"""Read out a conditional innovation separately from the exact linear branch."""
import numpy as np
from .exact_base_completion import ExactBaseFunctionalLearner,spectral
from .moments import ridge_readout


def innovation_readout(g,k,dimension,ridge):
    if ridge <= 0:raise ValueError('Positive regularization is required')
    phi=np.r_[np.arange(dimension),len(g)-1];hidden=np.arange(dimension,len(g)-1)
    a=g[np.ix_(phi,phi)];u=g[np.ix_(phi,hidden)]
    b=spectral(a)[1]@u
    raw=g[np.ix_(hidden,hidden)]-b.T@u-u.T@b+b.T@a@b
    residual=spectral(raw)[0]
    cross=k[hidden]-b.T@k[phi]
    base=ridge_readout(a,k[phi],ridge)
    correction=np.linalg.solve(residual+ridge*np.eye(len(hidden)),cross)
    result=np.zeros_like(k);result[phi]=base-b@correction;result[hidden]=correction
    diagnostics=dict(orthogonality_error=float(np.max(np.abs(u-a@b))),
                     residual_psd_adjustment=float(np.max(np.abs(raw-residual))),
                     base_weight_norm=float(np.linalg.norm(base)),correction_weight_norm=float(np.linalg.norm(correction)))
    return result,diagnostics


class OrthogonalCompletionLearner(ExactBaseFunctionalLearner):
    def __init__(self,*args,orthogonal=True,**kwargs):
        super().__init__(*args,completion=True,**kwargs);self.orthogonal=orthogonal;self.innovation_diagnostics=None
    def observe(self,x,y):
        super().observe(x,y)
        if self.orthogonal:
            self.readout,self.innovation_diagnostics=innovation_readout(self.g,self.k,self.dimension,self.ridge)

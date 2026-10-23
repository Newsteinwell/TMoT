"""Conditional joint-moment completion preserving fixed input/target blocks."""
import numpy as np
from .moments import ClassificationMoments,RegressionMoments,ridge_readout
from .replay_functional import ReplayFunctionalLearner

def spectral(matrix):
    values,vectors=np.linalg.eigh((matrix+matrix.T)/2)
    threshold=max(float(np.max(np.abs(values),initial=0))*1e-12,1e-14)
    values=np.where(values>threshold,values,0.)
    inverse=np.divide(1.,values,out=np.zeros_like(values),where=values>0)
    return (vectors*values)@vectors.T,(vectors*inverse)@vectors.T,(vectors*np.sqrt(values))@vectors.T,(vectors*np.sqrt(inverse))@vectors.T

def complete_joint(g,k,exact_g,exact_k,exact_v,dimension):
    phi=np.r_[np.arange(dimension),len(g)-1];hidden=np.arange(dimension,len(g)-1)
    a=g[np.ix_(phi,phi)];u=g[np.ix_(phi,hidden)];a_inv=spectral(a)[1]
    b=a_inv@u
    residual_h=spectral(g[np.ix_(hidden,hidden)]-u.T@a_inv@u)[0]
    residual_cross=k[hidden]-u.T@a_inv@k[phi]
    exact_inv=spectral(exact_g)[1]
    residual_y=spectral(exact_v-exact_k.T@exact_inv@exact_k)[0]
    _,_,hroot,hinvroot=spectral(residual_h);_,_,yroot,yinvroot=spectral(residual_y)
    left,values,right=np.linalg.svd(hinvroot@residual_cross@yinvroot,full_matrices=False)
    cross=hroot@((left*np.minimum(values,1.))@right)@yroot
    result_g=np.zeros_like(g);result_k=np.zeros_like(k)
    result_g[np.ix_(phi,phi)]=exact_g
    result_g[np.ix_(phi,hidden)]=exact_g@b
    result_g[np.ix_(hidden,phi)]=b.T@exact_g
    result_g[np.ix_(hidden,hidden)]=b.T@exact_g@b+residual_h
    result_k[phi]=exact_k;result_k[hidden]=b.T@exact_k+cross
    joint=np.block([[result_g,result_k],[result_k.T,exact_v]])
    eigen=np.linalg.eigvalsh((joint+joint.T)/2)
    diagnostics=dict(clipped_singular_values=int(np.sum(values>1)),max_unclipped_correlation=float(values.max(initial=0)),
                     joint_min_eigenvalue=float(eigen.min()),joint_max_eigenvalue=float(eigen.max()),
                     input_block_error=float(np.max(np.abs(result_g[np.ix_(phi,phi)]-exact_g))),
                     target_cross_error=float(np.max(np.abs(result_k[phi]-exact_k))))
    if eigen.min() < -1e-7*max(1.,float(eigen.max())):raise ValueError('Completed joint moment is not numerically PSD')
    return result_g,result_k,diagnostics

class ExactBaseFunctionalLearner(ReplayFunctionalLearner):
    def __init__(self,*args,completion=True,**kwargs):
        super().__init__(*args,**kwargs);self.completion=completion;self.exact_memory=None;self.completion_diagnostics=None
    def observe(self,x,y):
        x=np.asarray(x,np.float32);y=np.asarray(y)
        if self.task=='regression' and y.ndim==1:y=y[:,None]
        if self.exact_memory is None:self.exact_memory=ClassificationMoments(self.dimension) if self.task=='classification' else RegressionMoments(self.dimension,y.shape[1])
        self.exact_memory.update(x,y)
        super().observe(x,y)
        if self.completion:
            g,k,_=self.exact_memory.risk_moments(self.classes)
            v=np.eye(len(self.classes))/len(self.classes) if self.task=='classification' else self.exact_memory.target_gram/self.exact_memory.n
            self.g,self.k,self.completion_diagnostics=complete_joint(self.g,self.k,g,k,v,self.dimension)
            self.readout=ridge_readout(self.g,self.k,self.ridge)
    @property
    def persistent_bytes(self):return super().persistent_bytes+self.exact_memory.nbytes

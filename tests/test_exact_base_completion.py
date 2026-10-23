import numpy as np
import torch
from tmot.moments import augment
from tmot.exact_base_completion import complete_joint,ExactBaseFunctionalLearner
from tmot.replay_functional import ReplayFunctionalLearner

def test_matching_empirical_joint_is_recovered():
    rng=np.random.default_rng(9);x=rng.normal(size=(300,4));h=np.c_[np.sin(x[:,0]),x[:,1]**2];y=(x[:,0]+h[:,0]+rng.normal(size=300))[:,None]
    z=augment(np.c_[x,h]);phi=augment(x);g=z.T@z/len(x);k=z.T@y/len(x);v=y.T@y/len(x)
    ng,nk,d=complete_joint(g,k,phi.T@phi/len(x),phi.T@y/len(x),v,4)
    np.testing.assert_allclose(ng,g,atol=1e-9);np.testing.assert_allclose(nk,k,atol=1e-9)
    assert d['joint_min_eigenvalue']>-1e-9

def test_extreme_cross_covariance_is_capped_and_exact_blocks_preserved():
    rng=np.random.default_rng(2);z=augment(rng.normal(size=(100,6)));y=rng.normal(size=(100,2));phi=np.r_[0:4,6]
    g=z.T@z/100;k=z.T@y/100;k[4:6]*=1000;eg=g[np.ix_(phi,phi)];ek=k[phi];v=y.T@y/100
    ng,nk,d=complete_joint(g,k,eg,ek,v,4)
    np.testing.assert_array_equal(ng[np.ix_(phi,phi)],eg);np.testing.assert_array_equal(nk[phi],ek)
    assert d['clipped_singular_values']>0 and d['joint_min_eigenvalue']>-1e-8

def test_disabled_path_and_class_growth():
    torch.set_num_threads(1);rng=np.random.default_rng(3);x=rng.normal(size=(24,4));y=np.repeat([2,8],12)
    old=ReplayFunctionalLearner(4,epochs=1,device='cpu',replay_count=32)
    control=ExactBaseFunctionalLearner(4,epochs=1,device='cpu',replay_count=32,completion=False)
    fixed=ExactBaseFunctionalLearner(4,epochs=1,device='cpu',replay_count=32)
    for m in (old,control,fixed):m.observe(x[:12],y[:12]);m.observe(x[12:],y[12:])
    np.testing.assert_array_equal(old.predict(x),control.predict(x));assert fixed.classes==[2,8]
    eg,ek,_=fixed.exact_memory.risk_moments(fixed.classes);idx=np.r_[0:4,len(fixed.g)-1]
    np.testing.assert_array_equal(fixed.g[np.ix_(idx,idx)],eg);np.testing.assert_array_equal(fixed.k[idx],ek)

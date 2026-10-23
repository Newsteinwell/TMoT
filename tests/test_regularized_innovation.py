import numpy as np
import torch
from tmot.moments import augment,ridge_readout
from tmot.regularized_innovation import regularized_innovation_readout,RegularizedInnovationLearner
from tmot.exact_base_completion import ExactBaseFunctionalLearner


def test_explicit_regularized_feature_and_target_residual_fit():
    rng=np.random.default_rng(11);x=rng.normal(size=(90,4));phi=augment(x)
    h=np.c_[x[:,0]**2+x[:,1],np.sin(x[:,1])];y=np.c_[x[:,0]+h[:,0],x[:,2]-h[:,1]]
    z=augment(np.c_[x,h]);g=z.T@z/len(x);k=z.T@y/len(x);ridge=.3
    result,_=regularized_innovation_readout(g,k,4,ridge)
    a=phi.T@phi/len(x);base=ridge_readout(a,phi.T@y/len(x),ridge)
    b=ridge_readout(a,phi.T@h/len(x),ridge);res=h-phi@b;target=y-phi@base
    correction=np.linalg.solve(res.T@res/len(x)+ridge*np.eye(2),res.T@target/len(x))
    np.testing.assert_allclose(z@result,phi@base+res@correction,atol=1e-11)


def test_rank_deficient_features_and_weak_direction_stability():
    rng=np.random.default_rng(7);x=rng.normal(size=(20,40));x[:,0]*=1e-9;x[:,1]=x[:,2]
    h=rng.normal(size=(20,3));y=rng.normal(size=(20,2));z=augment(np.c_[x,h]);g=z.T@z/20;k=z.T@y/20
    w,d=regularized_innovation_readout(g,k,40,.01)
    assert np.isfinite(w).all() and np.linalg.norm(w)<100
    altered=g.copy();altered[0,0]+=1e-14
    w2,_=regularized_innovation_readout(altered,k,40,.01)
    np.testing.assert_allclose(w,w2,atol=1e-8,rtol=1e-8)


def test_zero_hidden_equals_base_and_class_growth_control():
    torch.set_num_threads(1);rng=np.random.default_rng(2);x=rng.normal(size=(24,4));phi=augment(x);y=np.repeat([3,9],12)
    z=augment(np.c_[x,np.zeros((24,2))]);target=rng.normal(size=(24,2))
    w,_=regularized_innovation_readout(z.T@z/24,z.T@target/24,4,.1)
    np.testing.assert_allclose(z@w,phi@ridge_readout(phi.T@phi/24,phi.T@target/24,.1),atol=1e-12)
    old=ExactBaseFunctionalLearner(4,epochs=1,device='cpu',replay_count=32)
    control=RegularizedInnovationLearner(4,epochs=1,device='cpu',replay_count=32,regularized=False)
    new=RegularizedInnovationLearner(4,epochs=1,device='cpu',replay_count=32)
    for model in (old,control,new):model.observe(x[:12],y[:12]);model.observe(x[12:],y[12:])
    np.testing.assert_array_equal(old.readout,control.readout)
    np.testing.assert_array_equal(old.g,new.g);assert new.classes==[3,9]

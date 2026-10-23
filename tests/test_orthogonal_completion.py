import numpy as np
import torch
from tmot.moments import augment,ridge_readout
from tmot.orthogonal_completion import innovation_readout,OrthogonalCompletionLearner
from tmot.exact_base_completion import ExactBaseFunctionalLearner


def test_empirical_innovation_agrees_with_explicit_residualized_fit():
    rng=np.random.default_rng(51);x=rng.normal(size=(200,4));phi=augment(x)
    h=np.c_[x[:,0]**2,np.sin(x[:,1])];y=np.c_[x[:,0]+h[:,0],x[:,2]-h[:,1]]
    z=augment(np.c_[x,h]);g=z.T@z/len(x);k=z.T@y/len(x);ridge=.1
    w,d=innovation_readout(g,k,4,ridge)
    b=np.linalg.lstsq(phi,h,rcond=None)[0];res=h-phi@b
    base=ridge_readout(phi.T@phi/len(x),phi.T@y/len(x),ridge)
    wh=np.linalg.solve(res.T@res/len(x)+ridge*np.eye(2),res.T@y/len(x))
    np.testing.assert_allclose(z@w,phi@base+res@wh,atol=1e-10)
    assert d['orthogonality_error']<1e-10


def test_linear_duplicate_functions_do_not_change_base_prediction():
    rng=np.random.default_rng(2);x=rng.normal(size=(120,3));phi=augment(x);h=phi@rng.normal(size=(4,2));y=rng.normal(size=(120,2))
    z=augment(np.c_[x,h]);w,_=innovation_readout(z.T@z/120,z.T@y/120,3,.3)
    np.testing.assert_allclose(z@w,phi@ridge_readout(phi.T@phi/120,phi.T@y/120,.3),atol=1e-10)


def test_disabled_control_and_arriving_classes():
    torch.set_num_threads(1);rng=np.random.default_rng(23);x=rng.normal(size=(24,4));y=np.repeat([3,9],12)
    old=ExactBaseFunctionalLearner(4,epochs=1,device='cpu',replay_count=32)
    control=OrthogonalCompletionLearner(4,epochs=1,device='cpu',replay_count=32,orthogonal=False)
    new=OrthogonalCompletionLearner(4,epochs=1,device='cpu',replay_count=32)
    for model in (old,control,new):
        model.observe(x[:12],y[:12]);model.observe(x[12:],y[12:])
    np.testing.assert_array_equal(old.readout,control.readout)
    np.testing.assert_array_equal(old.g,new.g);assert new.classes==[3,9]
    assert np.isfinite(new.readout).all()

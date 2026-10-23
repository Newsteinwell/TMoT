import numpy as np
import torch
from tmot.nonlinear_moments import NonlinearMomentLearner,sample_cell
from tmot.moments import Cell
from tmot.gradient_diagnostic import risk_gradient,CachedGaussian,compare_gradients

def test_cached_gaussian_matches_original_draw_exactly():
    x=np.random.default_rng(3).normal(size=(30,5));cell=Cell.from_x(x)
    np.testing.assert_array_equal(CachedGaussian(cell).sample(100,np.random.default_rng(4)),sample_cell(cell,100,np.random.default_rng(4)))

def test_gradient_chunking_and_finite_difference():
    torch.set_num_threads(1);rng=np.random.default_rng(4);x=rng.normal(size=(20,4));y=rng.normal(size=(20,2))
    m=NonlinearMomentLearner(4,'regression',device='cpu',epochs=1,width=8)
    m.observe(x,y)
    loss,g=risk_gradient(m,x,y,7);other,h=risk_gradient(m,x,y,100)
    np.testing.assert_allclose(g,h,rtol=2e-5,atol=1e-7);assert abs(loss-other)<1e-6
    p=list(m.net.parameters())[-1];eps=.001
    with torch.no_grad():p[0]+=eps
    plus,_=risk_gradient(m,x,y)
    with torch.no_grad():p[0]-=2*eps
    minus,_=risk_gradient(m,x,y)
    with torch.no_grad():p[0]+=eps
    np.testing.assert_allclose((plus-minus)/(2*eps),g[-2],rtol=.01,atol=1e-4)
    assert compare_gradients(g,g)['cosine']>.999999

def test_macro_gradient_invariant_to_class_replication():
    torch.set_num_threads(1);rng=np.random.default_rng(7)
    x=rng.normal(size=(12,4));y=np.repeat([3,8],6)
    m=NonlinearMomentLearner(4,device='cpu',epochs=1,width=8,loss='task');m.observe(x,y)
    loss,g=risk_gradient(m,x,y)
    ids=np.r_[np.tile(np.arange(6),3),np.arange(6,12)]
    other,h=risk_gradient(m,x[ids],y[ids])
    np.testing.assert_allclose(g,h,rtol=2e-5,atol=1e-7);assert abs(loss-other)<1e-6

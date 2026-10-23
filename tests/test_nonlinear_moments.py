import numpy as np
from tmot.moments import Cell, augment, ridge_readout
from tmot.nonlinear_moments import sample_cell, NonlinearMomentLearner


def test_gaussian_replay_joint_mean_and_shrunk_covariance():
    rng=np.random.default_rng(9);x=rng.normal(size=(80,3));y=x[:,0:1]*2+x[:,1:2]
    joint=np.c_[x,y];cell=Cell.from_x(joint)
    generated=sample_cell(cell,80000,rng)
    cov=np.cov(joint,rowvar=False,bias=True);expected=.9*cov+.1*np.diag(np.diag(cov))
    np.testing.assert_allclose(generated.mean(0),joint.mean(0),atol=.025)
    np.testing.assert_allclose(np.cov(generated,rowvar=False,bias=True),expected,atol=.06)


def test_zero_residual_equals_cumulative_ridge_and_no_samples_retained():
    rng=np.random.default_rng(3);x=rng.normal(size=(30,4)).astype('float32');y=rng.normal(size=(30,2))
    m=NonlinearMomentLearner(4,'regression',device='cpu',epochs=1,replay_count=32,width=8,alpha=0.)
    m.observe(x[:15],y[:15]);m.observe(x[15:],y[15:])
    g,k,_=m.memory.risk_moments();expected=augment(x)@ridge_readout(g,k,.1)
    np.testing.assert_allclose(m.predict(x),expected,atol=1e-8)
    assert m.buffer is None and len(m.joint_cells)==2 and m.seen==30


def test_class_growth_replay_only_arrived_classes_and_er_budget():
    rng=np.random.default_rng(5);x=rng.normal(size=(24,4)).astype('float32');y=np.repeat([9,2,7],8)
    for mode in ('gaussian','er'):
        m=NonlinearMomentLearner(4,device='cpu',epochs=1,replay_count=32,width=8,mode=mode,replay_bytes=256)
        m.observe(x[:16],y[:16]);hx,hy=m._historical(None)
        assert set(hy)<=set([2,9])
        m.observe(x[16:],y[16:]);assert set(m.classes)==set([2,7,9])
        assert m.predict(x).shape==(24,)
        if mode=='er':assert sum(v.nbytes for v in m.buffer)+m.keys.nbytes<=256

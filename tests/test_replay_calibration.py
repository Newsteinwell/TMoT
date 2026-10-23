import numpy as np
import pytest
from tmot.moments import RegressionMoments
from tmot.replay_calibration import calibrate_regression
from tmot.nonlinear_moments import NonlinearMomentLearner


def test_finite_batch_matches_moments_including_singular_targets():
    rng=np.random.default_rng(7);x=rng.normal(size=(70,12));y=np.c_[x[:,0]+x[:,1],2*(x[:,0]+x[:,1]),np.ones(70)]
    memory=RegressionMoments(12,3);memory.update(x,y)
    sx=rng.normal(size=(120,12))*3+2;sy=rng.normal(size=(120,3))
    cx,cy,stats=calibrate_regression(sx,sy,memory,rng)
    expected=np.cov(np.c_[x,y],rowvar=False,bias=True);actual=np.cov(np.c_[cx,cy],rowvar=False,bias=True)
    np.testing.assert_allclose(cx.mean(0),x.mean(0),atol=1e-6)
    np.testing.assert_allclose(cy.mean(0),y.mean(0),atol=1e-6)
    np.testing.assert_allclose(actual[:12,12:],expected[:12,12:],atol=1e-6)
    np.testing.assert_allclose(actual[12:,12:],expected[12:,12:],atol=1e-6)
    np.testing.assert_allclose(np.diag(actual),np.diag(expected),atol=1e-6)
    assert stats['after_cross_error']<1e-6


def test_degenerate_generated_batch_and_rank_limit():
    rng=np.random.default_rng(9);x=rng.normal(size=(30,4));y=x[:,:2]
    memory=RegressionMoments(4,2);memory.update(x,y)
    cx,cy,stats=calibrate_regression(np.zeros((50,4)),np.zeros((50,2)),memory,rng)
    assert np.isfinite(cx).all() and stats['after_cross_error']<1e-6
    with pytest.raises(ValueError):calibrate_regression(np.zeros((3,4)),np.zeros((3,2)),memory,rng)


def test_calibration_sees_only_history_before_current_update():
    rng=np.random.default_rng(3);x=rng.normal(size=(32,4)).astype('float32');y=x[:,:1]**2
    model=NonlinearMomentLearner(4,'regression',mode='local',device='cpu',epochs=1,width=8,replay_count=64,
                                preserve_targets=True,calibrate_replay=True)
    model.observe(x[:16],y[:16]);model.observe(x[16:],y[16:])
    assert model.last_calibration['history_samples']==16
    assert model.memory.n==32 and model.last_calibration['after_cross_error']<1e-5

import numpy as np
from tmot.local_moments import LocalCell,LocalMomentMixture
from tmot.nonlinear_moments import NonlinearMomentLearner,sample_cell
from tmot.moments import Cell


def test_low_rank_cell_preserves_marginal_variance():
    rng=np.random.default_rng(8);x=rng.normal(size=(100,12))@rng.normal(size=(12,12))
    cell=LocalCell.fit(x,rank=3)
    np.testing.assert_allclose(np.sum(cell.factor**2,axis=1)+cell.residual_sd**2,x.var(0),rtol=2e-6)
    assert cell.factor.shape==(12,3)


def test_local_joint_moments_better_preserve_bimodal_nonlinear_relation():
    rng=np.random.default_rng(6);x=np.r_[rng.normal(-3,.15,160),rng.normal(3,.15,160)][:,None];y=x*x
    model=LocalMomentMixture(1,'regression',components=2);model.update(x,y)
    sx,sy=model.sample(10000,[],rng)
    global_samples=sample_cell(Cell.from_x(np.c_[x,y]),10000,rng)
    local_error=np.mean((sy-sx*sx)**2)
    global_error=np.mean((global_samples[:,1:]-global_samples[:,:1]**2)**2)
    assert local_error<.25*global_error
    assert sum(c.n for c in model.cells[None])==320


def test_class_growth_and_regression_local_replay_smoke():
    rng=np.random.default_rng(1);x=rng.normal(size=(40,4)).astype('float32')
    for task,y in [('classification',np.repeat([9,4],20)),('regression',x[:,:1]**2)]:
        m=NonlinearMomentLearner(4,task,mode='local',device='cpu',epochs=1,width=8,replay_count=32)
        m.observe(x[:20],y[:20]);m.observe(x[20:],y[20:])
        assert m.buffer is None and np.isfinite(m.predict(x)).all()
        assert m.historical_training_samples==32
        assert sum(c.n for cells in m.local.cells.values() for c in cells)==40

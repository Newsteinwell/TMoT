import numpy as np
from tmot.local_moments import LocalCell
from tmot.local_experts import LocalExpert,GatedLocalLearner
from tmot.wimot_core.readout import density


def test_woodbury_predictions_and_density_match_dense_and_wimot():
    rng=np.random.default_rng(8);data=rng.normal(size=(50,10));cell=LocalCell.fit_target_preserving(data,8,rank=3)
    expert=LocalExpert(cell,8,.1);factor=cell.factor[:8].astype(float)
    cov=factor@factor.T+np.diag(1/expert.invdiag);precision=np.linalg.inv(cov)
    x=rng.normal(size=(12,8));cross=factor@cell.factor[8:].astype(float).T
    expected=cell.mean[8:]+(x-cell.mean[:8])@precision@cross
    np.testing.assert_allclose(expert.conditional_mean(x),expected,rtol=1e-9,atol=1e-9)
    reference=dict(center=np.zeros(8),transform=np.eye(8),means=cell.mean[:8][None],precisions=precision[None],logdets=np.array([np.linalg.slogdet(cov)[1]]))
    for df in (None,10.):np.testing.assert_allclose(expert.log_density(x,df),density(x,reference,df)[:,0],atol=1e-9)


def test_class_growth_and_multitarget_outputs_have_finite_normalized_gates():
    rng=np.random.default_rng(9);x=rng.normal(size=(48,6));y=np.repeat([9,2,7],16)
    model=GatedLocalLearner(6,components=1)
    model.observe(x[:32],y[:32]);assert set(model.predict(x))<=set([2,9])
    model.observe(x[32:],y[32:]);np.testing.assert_allclose(model.probabilities(x).sum(1),1,atol=1e-12)
    assert len(model.classes)==3
    model=GatedLocalLearner(6,'regression',components=1)
    model.observe(x[:24],x[:24,:2]**2);model.observe(x[24:],x[24:,:2]**2)
    assert model.predict(x).shape==(48,2) and np.isfinite(model.predict(x)).all()


def test_large_prior_strength_recovers_global_readout_and_expands_classes():
    from tmot.local_experts import PriorGatedLocalLearner
    from tmot.moments import augment
    rng=np.random.default_rng(17);x=rng.normal(size=(60,5));y=rng.normal(size=(60,2))
    model=PriorGatedLocalLearner(5,'regression',ridge=1e10,components=1)
    model.observe(x[:30],y[:30]);model.observe(x[30:],y[30:])
    np.testing.assert_allclose(model.predict(x),augment(x)@model.global_readout,atol=1e-7)
    model=PriorGatedLocalLearner(5,ridge=1.,components=1)
    model.observe(x[:30],np.repeat([4,9],15));model.observe(x[30:],np.repeat([7,2],15))
    assert model.global_readout.shape==(6,4)
    assert set(model.predict(x))<=set([2,4,7,9])


def test_prior_expert_satisfies_centered_dense_normal_equation():
    from tmot.local_experts import PriorGatedLocalLearner
    from tmot.moments import augment
    rng=np.random.default_rng(29);x=rng.normal(size=(40,5));y=x[:,:2]**2
    model=PriorGatedLocalLearner(5,'regression',ridge=.3,components=1);model.observe(x,y)
    expert=model.experts[0];cell=model.local.cells[None][0]
    f=cell.factor.astype(float);cov=f[:5]@f[:5].T+np.diag(cell.residual_sd[:5].astype(float)**2)
    cross=f[:5]@f[5:].T
    expected=np.linalg.solve(cov+expert.regularization*np.eye(5),cross+expert.regularization*model.global_readout[:-1])
    np.testing.assert_allclose(expert.beta,expected,rtol=1e-8,atol=1e-8)
    center=(cell.mean[5:]+.3*(augment(cell.mean[:5][None])@model.global_readout)[0])/1.3
    np.testing.assert_allclose(expert.target_mean,center,atol=1e-9)

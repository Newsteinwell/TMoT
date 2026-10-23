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

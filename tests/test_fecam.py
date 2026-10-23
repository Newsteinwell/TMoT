import numpy as np
from tmot.fecam import FeCAMLearner,unit_rows


def test_matches_official_pretrained_head_formula_and_repeated_class_merge():
    rng=np.random.default_rng(4);x=rng.normal(size=(60,8));y=np.tile([4,9,2],20);query=rng.normal(size=(11,8))
    model=FeCAMLearner(8)
    for rows in np.array_split(np.arange(60),3):model.observe(x[rows],y[rows])
    expected=[];normalized=unit_rows(x)
    for c in model.classes:
        data=normalized[y==c];mean=data.mean(0);cov=np.cov(data.T)
        cov+=np.eye(8)*np.diag(cov).mean()
        sd=np.sqrt(np.diag(cov));cov/=np.outer(sd,sd)
        delta=unit_rows(query)-mean
        expected.append(np.diag(delta@np.linalg.pinv(cov)@delta.T))
    np.testing.assert_allclose(model.distances(query),np.array(expected).T,rtol=1e-10,atol=1e-10)


def test_singleton_zero_features_do_not_produce_nan_or_future_classes():
    model=FeCAMLearner(4);model.observe(np.zeros((1,4)),np.array([7]))
    assert np.isfinite(model.distances(np.ones((3,4)))).all()
    np.testing.assert_array_equal(model.predict(np.ones((3,4))),[7,7,7])

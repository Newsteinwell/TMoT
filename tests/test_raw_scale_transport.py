import numpy as np
from tmot.raw_scale_transport import RawScaleTransport,affine_diagonal_moments
from tmot.moments import augment,ridge_readout,RegressionMoments
from tmot.learner import MomentLearner


def test_diagonal_affine_moments_equal_explicit_dense_transform():
    rng=np.random.default_rng(8);x=rng.normal(size=(30,6));y=rng.normal(size=(30,2));z=augment(x);g=z.T@z/30;k=z.T@y/30
    scale=rng.uniform(.1,3,6);shift=rng.normal(size=6);h=np.eye(7);h[:-1,:-1]=np.diag(scale);h[:-1,-1]=shift
    tg,tk=affine_diagonal_moments(g,k,scale,shift)
    np.testing.assert_allclose(tg,h@g@h.T,atol=1e-12);np.testing.assert_allclose(tk,h@k,atol=1e-12)


def test_streaming_raw_scale_matches_reprocessing_all_arrived_rows():
    rng=np.random.default_rng(9);x=rng.normal(size=(32,6));x[16:,3:]*=10;y=x[:,:2]*.3+x[:,3:5]
    model=RawScaleTransport(6,3,task='regression');model.observe(x[:16],y[:16]);model.observe(x[16:],y[16:])
    np.testing.assert_array_equal(model.scale[:3],np.ones(3));np.testing.assert_array_equal(model.shift[:3],np.zeros(3))
    transformed=x*model.scale+model.shift;memory=RegressionMoments(6,2);memory.update(transformed,y)
    g,k,_=memory.risk_moments();expected=augment(transformed)@ridge_readout(g,k,.1)
    np.testing.assert_allclose(model.predict(x),expected,atol=1e-10)


def test_disabled_transport_reproduces_growing_class_ridge():
    rng=np.random.default_rng(11);x=rng.normal(size=(24,7)).astype('float32');y=np.repeat([2,5,9],8)
    a=RawScaleTransport(7,4,transport=False);b=MomentLearner(7,method='ridge',ridge=.1,device='cpu')
    for ids in (slice(0,16),slice(16,None)):
        a.observe(x[ids],y[ids]);b.observe(x[ids],y[ids])
        np.testing.assert_array_equal(a.readout,b.readout);np.testing.assert_array_equal(a.predict(x),b.predict(x))

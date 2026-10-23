import numpy as np
from tmot.stable_residual import StableSemanticBasis,StableResidualLearner
from tmot.moments import augment,ridge_readout

def test_zero_residual_is_exact_analytic_baseline():
    rng=np.random.default_rng(8);x=rng.normal(size=(24,8)).astype('float32');y=np.repeat([0,1,2,3],6)
    model=StableResidualLearner(8,alpha=0,device='cpu')
    model.observe(x[:12],y[:12]);model.observe(x[12:],y[12:])
    g,k,_=model.memory.risk_moments(model.classes)
    direct=np.array(model.classes)[(augment(x)@ridge_readout(g,k,.1)).argmax(1)]
    np.testing.assert_array_equal(model.predict(x),direct)

def test_residual_supports_regression_and_fixed_basis():
    rng=np.random.default_rng(9);x=rng.normal(size=(20,18)).astype('float32');y=x[:,:2]
    basis=StableSemanticBasis(2,dimension=16).fit(x[:10]);before=basis.transform(x[:10])
    model=StableResidualLearner(16,task='regression',epochs=1,device='cpu')
    model.observe(before,y[:10]);model.observe(basis.transform(x[10:]),y[10:])
    assert np.isfinite(model.predict(before)).all()
    np.testing.assert_array_equal(before,basis.transform(x[:10]))

import numpy as np
import pytest
import torch
from tmot.functional_basis import FunctionalBasis
from tmot.raw_temporal import RawCoordinateBasis
from tmot.moments import ClassificationMoments,ridge_readout,augment

def test_fixed_function_moments_equal_direct_full_risk():
    torch.set_num_threads(1);rng=np.random.default_rng(4);x=rng.normal(size=(30,5));y=np.repeat([2,7,9],10)
    base=RawCoordinateBasis(5).fit(x[:20]);f=FunctionalBasis(base,'classification',width=8,epochs=2,device='cpu').fit(x[:20],y[:20])
    before={k:v.clone() for k,v in f.net.state_dict().items()};z=f.transform(x)
    memory=ClassificationMoments(f.dimension)
    for ids in (np.arange(20),np.arange(20,30)):memory.update(f.transform(x[ids]),y[ids])
    g,k,_=memory.risk_moments([2,7,9]);w=ridge_readout(g,k,.1)
    direct=ClassificationMoments(f.dimension);direct.update(z,y);dg,dk,_=direct.risk_moments([2,7,9])
    np.testing.assert_allclose(augment(z)@w,augment(z)@ridge_readout(dg,dk,.1),atol=1e-8)
    np.testing.assert_array_equal(z[:,:5],base.transform(x).astype(np.float32))
    for key,value in f.net.state_dict().items():torch.testing.assert_close(value,before[key],atol=0,rtol=0)
    with pytest.raises(RuntimeError):f.fit(x,y)

def test_random_determinism_and_vector_regression():
    torch.set_num_threads(1);rng=np.random.default_rng(8);x=rng.normal(size=(12,5));y=rng.normal(size=(12,2))
    base=RawCoordinateBasis(5).fit(x)
    a=FunctionalBasis(base,'regression',mode='random',width=8,device='cpu').fit(x,y)
    b=FunctionalBasis(base,'regression',mode='random',width=8,device='cpu').fit(x,y+100)
    np.testing.assert_array_equal(a.transform(x),b.transform(x));assert a.optimizer_steps==0
    c=FunctionalBasis(base,'regression',width=8,epochs=2,device='cpu').fit(x,y)
    assert c.transform(x).shape==(12,13) and c.optimizer_steps==2

def test_explicit_first_calibration_preserves_default_and_can_stay_fixed():
    torch.set_num_threads(1);rng=np.random.default_rng(13);x=rng.normal(size=(24,5));y=rng.normal(size=(24,2))
    base=RawCoordinateBasis(5).fit(x[:12])
    a=FunctionalBasis(base,'regression',width=8,epochs=2,device='cpu').fit(x[:12],y[:12])
    b=FunctionalBasis(base,'regression',width=8,epochs=2,device='cpu').fit(x[:12],y[:12],calibration_x=x[:12])
    np.testing.assert_array_equal(a.transform(x),b.transform(x))
    c=FunctionalBasis(base,'regression',width=8,epochs=2,device='cpu').fit(x,y,calibration_x=x[:12])
    np.testing.assert_array_equal(a.mean,c.mean);np.testing.assert_array_equal(a.sd,c.sd)

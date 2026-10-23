import numpy as np
import pytest
from tmot.raw_temporal import raw_temporal,RawAppendedBasis,RawCoordinateBasis
from tmot.stable_residual import StableSemanticBasis


def test_raw_windows_preserve_original_units_and_sample_independence():
    x=np.arange(64,dtype=np.float32).reshape(2,2,16);d=raw_temporal(x)
    assert d.shape==(2,18)
    np.testing.assert_array_equal(d[:,-2:],x[:,:,-1])
    np.testing.assert_array_equal(d[0,:8],x[0,0].reshape(8,2).mean(1))
    np.testing.assert_array_equal(d[:1],raw_temporal(x[:1]))
    assert np.isfinite(raw_temporal(np.ones((2,3,1)))).all()


def test_raw_append_keeps_original_representation_and_affine_reconstruction():
    rng=np.random.default_rng(7);old=rng.normal(size=(20,25)).astype('float32');raw=rng.normal(size=(20,9)).astype('float32')*4+10
    x=np.concatenate([old[:,:-6],raw,old[:,-6:]],axis=1)
    base=StableSemanticBasis(6,32).fit(old[:10]);model=RawAppendedBasis(6,9,32).fit(x[:10]);z=model.transform(x)
    np.testing.assert_array_equal(z[:,:32],base.transform(old))
    np.testing.assert_allclose(z[:,32:]*model.raw.scale+model.raw.mean,raw,atol=2e-6)
    with pytest.raises(RuntimeError):model.raw.fit(raw)


def test_separate_normalized_shape_and_scale_do_not_linearly_encode_product():
    z=np.array([-1,-1,1,1.]);sd=np.array([1,2,1,2.]);features=np.c_[z,sd,np.ones(4)];target=z*sd
    residual=target-features@np.linalg.lstsq(features,target,rcond=None)[0]
    assert np.linalg.norm(residual)>.5
    enriched=np.c_[features,target];fit=enriched@np.linalg.lstsq(enriched,target,rcond=None)[0]
    np.testing.assert_allclose(fit,target,atol=1e-12)

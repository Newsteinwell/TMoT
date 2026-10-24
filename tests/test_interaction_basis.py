import numpy as np
import pytest
from tmot.interaction_basis import InteractionBasis,lagged_products
from tmot.stable_residual import StableSemanticBasis

def test_sample_local_and_directional():
 rng=np.random.default_rng(1);x=rng.normal(size=(5,3,20)).astype('float32')
 np.testing.assert_array_equal(lagged_products(x)[:2],lagged_products(x[:2]))
 np.testing.assert_allclose(lagged_products(x*3+4),lagged_products(x),atol=1e-6)
 assert not np.allclose(lagged_products(x[:,:,::-1]),lagged_products(x))
 assert np.isfinite(lagged_products(np.ones((2,3,20)))).all()

def test_disabled_exact_and_first_arrival_frozen():
 rng=np.random.default_rng(2);raw=rng.normal(size=(6,3,20));f=rng.normal(size=(6,30)).astype('float32')
 b=InteractionBasis(3,False).fit(f[:3],raw[:3]);ref=StableSemanticBasis(6,512).fit(f[:3])
 np.testing.assert_array_equal(b.transform(f,raw),ref.transform(f))
 b=InteractionBasis(3,True).fit(f[:3],raw[:3]);before=b.mean.copy();z=b.transform(f,raw)
 assert z.shape==(6,576) and np.isfinite(z).all()
 np.testing.assert_array_equal(b.mean,before)
 with pytest.raises(RuntimeError):b.fit(f,raw)

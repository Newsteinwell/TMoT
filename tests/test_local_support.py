import numpy as np
import pytest
from tmot.local_moments import LocalMomentMixture

def test_support_changes_small_arrival_partition():
    rng=np.random.default_rng(2);x=rng.normal(size=(24,8));y=np.zeros(24,dtype=int)
    a=LocalMomentMixture(8,'classification',components=4,support=16)
    b=LocalMomentMixture(8,'classification',components=4,support=4)
    a.update(x,y);b.update(x,y)
    assert len(a.cells[0])==1 and len(b.cells[0])==4
    assert sum(c.n for c in b.cells[0])==24
    assert all(np.isfinite(c.factor).all() for c in b.cells[0])

def test_default_support_reproduces_explicit_and_target_protection():
    rng=np.random.default_rng(3);x=rng.normal(size=(24,8));y=rng.normal(size=(24,2))
    a=LocalMomentMixture(8,'regression',preserve_targets=True)
    b=LocalMomentMixture(8,'regression',preserve_targets=True,support=16)
    a.update(x,y);b.update(x,y)
    np.testing.assert_array_equal(a.cells[None][0].factor,b.cells[None][0].factor)
    c=LocalMomentMixture(8,'regression',preserve_targets=True,support=4);c.update(x,y)
    assert len(c.cells[None])==4
    sx,sy=c.sample(100,[],rng);assert sx.shape==(100,8) and sy.shape==(100,2)
    assert np.isfinite(sx).all() and np.isfinite(sy).all()
    with pytest.raises(ValueError):LocalMomentMixture(8,'regression',support=0)

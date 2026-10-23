import numpy as np
from tmot.local_moments import LocalCell,LocalMomentMixture


def covariance(cell):return cell.factor.astype(float)@cell.factor.astype(float).T+np.diag(cell.residual_sd.astype(float)**2)


def test_target_cross_and_variance_preserved_under_aggressive_input_compression():
    rng=np.random.default_rng(11);x=rng.normal(size=(120,30));y=x@rng.normal(size=(30,3))+rng.normal(size=(120,3))
    rows=np.c_[x,y];cell=LocalCell.fit_target_preserving(rows,30,rank=2)
    got=covariance(cell);expected=np.cov(rows,rowvar=False,bias=True)
    np.testing.assert_allclose(got[:30,30:],expected[:30,30:],atol=2e-6,rtol=2e-6)
    np.testing.assert_allclose(got[30:,30:],expected[30:,30:],atol=2e-5,rtol=2e-6)
    np.testing.assert_allclose(np.diag(got),np.diag(expected),atol=2e-5,rtol=2e-6)
    assert cell.factor.shape[1]==5


def test_singular_constant_and_singleton_targets_are_supported():
    rng=np.random.default_rng(4)
    for n in (1,40):
        x=rng.normal(size=(n,8));y=np.c_[x[:,0],2*x[:,0],np.ones(n)]
        cell=LocalCell.fit_target_preserving(np.c_[x,y],8,rank=1)
        assert np.isfinite(cell.sample(20,rng)).all()
        expected=np.cov(np.c_[x,y],rowvar=False,bias=True) if n>1 else np.zeros((11,11))
        np.testing.assert_allclose(covariance(cell)[:8,8:],expected[:8,8:],atol=1e-6)
        np.testing.assert_allclose(covariance(cell)[8:,8:],expected[8:,8:],atol=1e-6)


def test_mixture_merge_keeps_cross_moments_across_arrivals():
    rng=np.random.default_rng(22);x=rng.normal(size=(160,10));y=np.c_[x[:,0]**2,x[:,1]+x[:,2]]
    model=LocalMomentMixture(10,'regression',components=4,rank=2,preserve_targets=True)
    model.update(x[:80],y[:80]);model.update(x[80:],y[80:])
    cells=model.cells[None];mean=sum(c.n*c.mean.astype(float) for c in cells)/160
    raw=sum(c.n*(covariance(c)+np.outer(c.mean.astype(float),c.mean.astype(float))) for c in cells)/160
    got=raw-np.outer(mean,mean);expected=np.cov(np.c_[x,y],rowvar=False,bias=True)
    np.testing.assert_allclose(got[:10,10:],expected[:10,10:],atol=1e-6)
    np.testing.assert_allclose(got[10:,10:],expected[10:,10:],atol=1e-6)

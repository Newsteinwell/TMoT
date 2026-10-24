import numpy as np
import pytest
from tmot.full_feature_diagnostic import weighted_ridge_grid,unprojected_features
from tmot.stable_residual import StableSemanticBasis

@pytest.mark.parametrize('n,d',[(35,7),(7,35),(12,12)])
def test_primal_dual_matches_independent_augmented_normal_equations(n,d):
    rng=np.random.default_rng(81);x=rng.normal(size=(n,d))+3;y=rng.normal(size=(n,4));w=np.arange(1,n+1,dtype=float);w/=w.sum()
    result,_=weighted_ridge_grid(x,y,w,[.001,.1,10.]);phi=np.c_[x,np.ones(n)]
    for ridge,(coef,bias) in result.items():
        penalty=np.eye(d+1)*ridge;penalty[-1,-1]=0
        independent=np.linalg.solve(phi.T@(w[:,None]*phi)+penalty,phi.T@(w[:,None]*y))
        np.testing.assert_allclose(np.r_[coef,bias[None,:]],independent,rtol=1e-7,atol=1e-9)
    shifted,_=weighted_ridge_grid(x+100,y-50,w,[.1]);a,b=shifted[.1];c,e=result[.1]
    np.testing.assert_allclose((x+100)@a+b,x@c+e-50,atol=1e-10)


def test_full_coordinates_preserve_fixed_first_calibration():
    rng=np.random.default_rng(9);x=rng.normal(size=(30,14)).astype(np.float32);b=StableSemanticBasis(4,32).fit(x[:6]);mean=b.mean.copy()
    full=unprojected_features(b,x)
    np.testing.assert_allclose(full[:,:10]/np.sqrt(24/10),(x[:,:10]-b.mean)/b.scale,rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(full[:,10:]/np.sqrt(8/4),(x[:,10:]-b.amp_mean)/b.amp_sd,rtol=1e-6,atol=1e-6)
    np.testing.assert_array_equal(b.mean,mean)

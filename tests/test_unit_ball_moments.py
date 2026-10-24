import numpy as np
from tmot.unit_ball_moments import bound_hidden_moments

def test_congruence_preserves_base_psd_and_bounds_second_moment():
 rng=np.random.default_rng(13);a=rng.normal(size=(8,8));cov=a.T@a;mean=rng.normal(size=8);m,c,diag=bound_hidden_moments(mean,cov,17,5)
 assert diag['changed'];assert abs(diag['second_norm_after']-1)<1e-12
 np.testing.assert_array_equal(m[:5],mean[:5]);np.testing.assert_array_equal(c[:5,:5],cov[:5,:5]);assert np.linalg.eigvalsh(c).min()>-1e-12
 d=np.diag([1.]*5+[diag['contraction']]*3)
 np.testing.assert_allclose(c,d@cov@d,atol=1e-12)

def test_admissible_moments_are_exactly_unchanged_and_bound_idempotent():
 mean=np.zeros(5);cov=np.eye(5)*.01;m,c,diag=bound_hidden_moments(mean,cov,4,2)
 np.testing.assert_array_equal(m,mean);np.testing.assert_array_equal(c,cov);assert not diag['changed']
 m1,c1,_=bound_hidden_moments(mean,cov*100,4,2);m2,c2,_=bound_hidden_moments(m1,c1,4,2)
 np.testing.assert_allclose(m1,m2,atol=1e-14);np.testing.assert_allclose(c1,c2,atol=1e-14)


def test_diagnostics_serialize_for_active_and_inactive_contraction():
 import json
 for magnitude in (.01, 100.):
  _,_,diag=bound_hidden_moments(np.zeros(5),np.eye(5)*magnitude,4,2)
  restored=json.loads(json.dumps(diag))
  assert isinstance(restored['changed'],bool)
  assert restored==diag

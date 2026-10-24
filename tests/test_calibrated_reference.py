import numpy as np
import pytest
from tmot.calibrated_reference import CalibratedReference
from tmot.causal_reference import CausalReference

def test_known_slope_intercept_and_fixed_calibration():
 p=np.array([[1.],[2.],[3.],[4.]],dtype=np.float32);y=.25*p+7
 r=CalibratedReference().fit(p,y);np.testing.assert_allclose(r.beta,[.25]);assert r.calibration_count==4
 np.testing.assert_array_equal(r.features(np.ones((4,2)),p),CausalReference('append').fit(p).features(np.ones((4,2)),p))
 with pytest.raises(RuntimeError):r.fit(p,y*2)
 residual=y-r.offset(p);np.testing.assert_array_equal(residual+r.offset(p),y)

def test_zero_variance_and_clipped_extremes():
 p=np.arange(4.)[:,None]
 for target,beta in ((-p,0.),(3*p,1.),(np.ones_like(p),0.)):
  np.testing.assert_array_equal(CalibratedReference().fit(p,target).beta,[beta])
 np.testing.assert_array_equal(CalibratedReference().fit(np.ones_like(p),p).beta,[0.])

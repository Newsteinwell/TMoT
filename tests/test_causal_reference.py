import numpy as np
import pytest
from tmot.causal_reference import CausalReference


def test_only_first_arrival_calibrates_and_residual_roundtrip():
    past=np.array([[2.], [4.], [100.]])
    model=CausalReference('residual').fit(past[:2])
    np.testing.assert_array_equal(model.mean,[3.])
    np.testing.assert_array_equal(model.scale,[1.])
    x=np.zeros((3,2))
    np.testing.assert_array_equal(model.features(x,past)[:,-1],[-1,1,97])
    target=np.array([[6.],[8.],[99.]])
    residual=target-model.offset(past)
    np.testing.assert_array_equal(residual+model.offset(past),target)
    with pytest.raises(RuntimeError):model.fit(past)


def test_controls_share_features_but_not_prior():
    past=np.array([[1.],[3.]])
    x=np.ones((2,4))
    a=CausalReference('append').fit(past)
    b=CausalReference('residual').fit(past)
    np.testing.assert_array_equal(a.features(x,past),b.features(x,past))
    np.testing.assert_array_equal(a.offset(past),np.zeros_like(past))
    assert CausalReference('none').fit(past).features(x,past) is x

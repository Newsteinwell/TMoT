import numpy as np
from scripts.prepare_round3 import causal_window

def test_causal_fill_no_future_or_boundary_access():
    x=np.array([[1.,2.],[np.nan,3.],[4.,np.nan]])
    y=causal_window(x)
    np.testing.assert_array_equal(y[:,:2],[[1,2],[1,3],[4,3]])
    np.testing.assert_array_equal(y[:,2:],[[1,1],[0,1],[1,0]])
    assert causal_window(np.array([[np.nan,2.],[4.,3.]])) is None

import numpy as np
from tmot.hard_prequential import hard_weights

def test_first_stage_uniform_then_fixed_tie_order():
 np.testing.assert_allclose(hard_weights([0,0,0],0),[1/3]*3)
 np.testing.assert_array_equal(hard_weights([3,1,2],20),[0,1,0])
 np.testing.assert_array_equal(hard_weights([2,2,3],20),[1,0,0])
 np.testing.assert_array_equal(hard_weights([0,0,0],20),[1,0,0])

def test_global_loss_scale_does_not_change_selection():
 np.testing.assert_array_equal(hard_weights([5,3,1],2),hard_weights([500,300,100],2))

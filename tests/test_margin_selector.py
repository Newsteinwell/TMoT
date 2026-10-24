import numpy as np
import pytest
from tmot.margin_selector import relative_margin,select_by_margin

def test_scale_invariance_and_tie_favors_reference():
 r=np.array([[1.,3.,5.],[2.,2.,9.],[0.,0.,4.]])
 np.testing.assert_allclose(relative_margin(r),relative_margin(r*37))
 p,selected=select_by_margin(r,r*2,np.array([4,7,9]))
 np.testing.assert_array_equal(p,[4,4,4]);assert not selected.any()

def test_selects_margin_without_labels_or_accuracy_access():
 r=np.array([[1.,2.],[1.,4.]])
 c=np.array([[4.,1.],[2.,1.]])
 pred,choice=select_by_margin(r,c,[10,20]);np.testing.assert_array_equal(pred,[20,10]);np.testing.assert_array_equal(choice,[True,False])

def test_invalid_distances_rejected():
 for x in ([[1.]],[[1.,float('nan')]],[[1.,-1.]]):
  with pytest.raises(ValueError):relative_margin(x)

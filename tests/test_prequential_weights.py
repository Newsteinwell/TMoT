import numpy as np
from tmot.prequential_weights import InverseErrorWeights

def test_uniform_and_inverse_cumulative_errors():
 m=InverseErrorWeights(2);np.testing.assert_array_equal(m.weights(),[.5,.5])
 m.update(np.array([[[1.]],[[2.]]]),np.array([[0.]]));np.testing.assert_allclose(m.weights(),[.8,.2])
 m.update(np.array([[[3.]],[[0.]]]),np.array([[0.]]));np.testing.assert_allclose(m.weights(),[2/7,5/7]);assert m.count==2

def test_zero_loss_and_scale_invariance():
 m=InverseErrorWeights(3);m.update(np.array([[[0.]],[[0.]],[[1.]]]),np.array([[0.]]));np.testing.assert_array_equal(m.weights(),[.5,.5,0])
 a=InverseErrorWeights(2);b=InverseErrorWeights(2);p=np.array([[[2.],[3.]],[[4.],[6.]]]);y=np.array([[1.],[1.]])
 a.update(p,y);b.update(p*100,y*100);np.testing.assert_allclose(a.weights(),b.weights())

import unittest
import numpy as np
from tmot.neighbor_readout import NeighborReadout

class NeighborTests(unittest.TestCase):
    def test_growth_and_ties(self):
        m=NeighborReadout(1,'classification',2)
        m.observe([[0],[2]],np.array([7,3]));self.assertEqual(m.predict([[1]])[0],3)
        m.observe([[8],[9]],np.array([11,11]));self.assertEqual(m.predict([[8.5]])[0],11)
    def test_vector_regression(self):
        m=NeighborReadout(1,'regression',2)
        m.observe([[0],[2],[10]],np.array([[1.,2.],[3.,6.],[100.,200.]]))
        np.testing.assert_array_equal(m.predict([[1]]),[[2.,4.]])
    def test_reservoir_chunk_invariance_and_budget(self):
        x=np.arange(200).reshape(100,2);y=np.arange(100)
        a=NeighborReadout(2,'classification',3,160,42);b=NeighborReadout(2,'classification',3,160,42)
        a.observe(x,y)
        for start in range(0,100,10):b.observe(x[start:start+10],y[start:start+10])
        np.testing.assert_array_equal(a.x,b.x);np.testing.assert_array_equal(a.y,b.y)
        self.assertEqual(a.seen,100);self.assertLessEqual(a.persistent_bytes,160)
        self.assertEqual(len(a.x),10)

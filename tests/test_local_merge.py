import numpy as np
from tmot.local_moments import merge_small_cells,LocalMomentMixture

def test_merge_nearest_and_noop():
    x=np.array([[0.],[.1],[.2],[.3],[1.],[10.],[10.1],[10.2],[10.3]])
    labels=np.array([0,0,0,0,1,2,2,2,2])
    np.testing.assert_array_equal(merge_small_cells(x,labels,1),labels)
    merged=merge_small_cells(x,labels,4)
    np.testing.assert_array_equal(merged,[0,0,0,0,0,2,2,2,2])
    np.testing.assert_array_equal(labels,[0,0,0,0,1,2,2,2,2])

def test_tiny_arrival_and_ties_terminate():
    x=np.zeros((3,2));labels=np.arange(3)
    merged=merge_small_cells(x,labels,4)
    assert len(np.unique(merged))==1 and len(merged)==3

def test_arrival_counts_and_minimum_for_both_tasks():
    rng=np.random.default_rng(8);x=rng.normal(size=(32,6))
    for task,y in [('classification',np.repeat([1,9],16)),('regression',rng.normal(size=(32,2)))]:
        m=LocalMomentMixture(6,task,components=4,support=4,min_cell=4,preserve_targets=True)
        m.update(x,y)
        cells=[c for group in m.cells.values() for c in group]
        assert sum(c.n for c in cells)==32 and min(c.n for c in cells)>=4
        a,b=m.sample(100,[1,9] if task=='classification' else [],rng)
        assert np.isfinite(a).all() and np.isfinite(b).all()

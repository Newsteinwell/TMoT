import numpy as np
import pytest
from tmot.covariance_replay import CovarianceMixture
from tmot.empirical_cell_replay import EmpiricalMixture
@pytest.mark.parametrize('task',['classification','regression'])
def test_empirical_support_and_matched_stream(task):
    rng=np.random.default_rng(8);x=rng.normal(size=(40,5)).astype(np.float32);y=np.repeat([2,9],20) if task=='classification' else (x[:,:1]**2).astype(np.float32)
    args=dict(dimension=5,task=task,seed=7,preserve_targets=True,support=4,min_cell=4)
    old=CovarianceMixture(**args);control=EmpiricalMixture(**args);emp=EmpiricalMixture(**args,variant='empirical')
    for m in [old,control,emp]:m.update(x,y)
    streams=[np.random.default_rng(3) for _ in range(3)];labels=[2,9] if task=='classification' else []
    samples=[m.sample(127,labels,stream) for m,stream in zip([old,control,emp],streams)]
    for actual,expected in zip(samples[1],samples[0]):np.testing.assert_array_equal(actual,expected)
    assert all(stream.bit_generator.state==streams[0].bit_generator.state for stream in streams)
    if task=='classification':np.testing.assert_array_equal(samples[1][1],samples[2][1])
    ex,ey=samples[2]
    for row,target in zip(ex,ey):
        candidates=np.flatnonzero(np.all(x==row,axis=1));assert len(candidates);assert any(np.array_equal(y[i],target) for i in candidates)
    for label,cells in emp.cells.items():
        for cell,other in zip(cells,control.cells[label]):
            np.testing.assert_array_equal(cell.observed,other.observed);assert len(cell.observed)==cell.n

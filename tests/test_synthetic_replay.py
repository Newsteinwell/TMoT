import numpy as np
import pytest
import torch
from tmot.synthetic_replay import SyntheticMixture,fit_points
from tmot.covariance_replay import CovarianceMixture
@pytest.mark.parametrize('task',['classification','regression'])
def test_synthetic_cells(task):
 torch.set_num_threads(1);rng=np.random.default_rng(4);x=rng.normal(size=(40,5)).astype(np.float32);y=np.repeat([2,9],20) if task=='classification' else x[:,:1]**2
 args=dict(dimension=5,task=task,preserve_targets=True,support=4,min_cell=4)
 models=[CovarianceMixture(**args)]+[SyntheticMixture(**args,variant=v) for v in ['gaussian','finite','matched']]
 for m in models:m.update(x,y)
 streams=[np.random.default_rng(3) for _ in models];samples=[m.sample(129,[2,9] if task=='classification' else [],s) for m,s in zip(models,streams)]
 for a,b in zip(samples[0],samples[1]):np.testing.assert_array_equal(a,b)
 assert all(s.bit_generator.state==streams[0].bit_generator.state for s in streams)
 if task=='classification':
  for _,ys in samples:np.testing.assert_array_equal(ys,samples[0][1])
 for cells in models[-1].cells.values():
  for c in cells:assert c.points.shape[0]==16 and c.fitting['final']<=c.fitting['initial'];assert not hasattr(c,'observed')
 assert any(c.fitting['final']<c.fitting['initial']*.8 for cells in models[-1].cells.values() for c in cells)
 assert models[-1].nbytes==models[-2].nbytes

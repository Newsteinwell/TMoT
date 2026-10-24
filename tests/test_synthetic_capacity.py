import numpy as np
import pytest
from tmot.task_synthetic_replay import TaskSyntheticMixture
@pytest.mark.parametrize('task',['classification','regression'])
def test_capacity_stream_and_memory(task):
 rng=np.random.default_rng(3);x=rng.normal(size=(40,5)).astype(np.float32);y=np.repeat([2,9],20) if task=='classification' else x[:,:1]
 args=dict(dimension=5,task=task,variant='finite',preserve_targets=True,support=4,min_cell=4)
 models=[TaskSyntheticMixture(**args,points_per_cell=n) for n in (16,64,256)]
 for m in models:m.update(x,y)
 streams=[np.random.default_rng(7) for _ in models];samples=[m.sample(200,[2,9] if task=='classification' else [],r) for m,r in zip(models,streams)]
 assert all(r.bit_generator.state==streams[0].bit_generator.state for r in streams)
 if task=='classification':
  for _,ys in samples:np.testing.assert_array_equal(ys,samples[0][1])
 count=sum(len(c) for c in models[0].cells.values());width=5+(task=='regression')
 for m,n in zip(models,(16,64,256)):
  assert all(c.points.shape==(n,width) for cs in m.cells.values() for c in cs)
  assert m.nbytes-models[0].nbytes==count*(n-16)*width*4

import numpy as np
import torch
import pytest
from tmot.synthetic_replay import SyntheticReplayLearner
from tmot.task_synthetic_replay import TaskSyntheticLearner
@pytest.mark.parametrize('task',['classification','regression'])
def test_control_hook_and_task_fit(task):
 torch.set_num_threads(1);rng=np.random.default_rng(8);x=rng.normal(size=(30,5)).astype(np.float32);y=np.repeat([1,7],15) if task=='classification' else x[:,:1]**2
 kwargs=dict(dimension=5,task=task,epochs=1,device='cpu',replay_count=32)
 a=SyntheticReplayLearner(**kwargs,variant='matched');b=TaskSyntheticLearner(**kwargs,variant='random');c=TaskSyntheticLearner(**kwargs,variant='task')
 for ids in [slice(0,15),slice(15,30)]:
  for m in [a,b,c]:m.observe(x[ids],y[ids])
  np.testing.assert_array_equal(a.readout,b.readout)
  assert b.local.teacher is None and c.local.teacher is None
 for cells in c.local.cells.values():
  for cell in cells:assert cell.fitting['final']<=cell.fitting['initial'] and cell.points.shape[0]==16
 assert any(cell.fitting['final']<cell.fitting['initial']*.8 for cells in c.local.cells.values() for cell in cells)
 assert b.persistent_bytes==c.persistent_bytes

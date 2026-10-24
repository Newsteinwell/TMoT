import copy
import numpy as np
import pytest
import torch
from tmot.task_synthetic_replay import TaskSyntheticLearner
from tmot.synthetic_drift_diagnostic import DriftDiagnostic
@pytest.mark.parametrize('task',['classification','regression'])
def test_read_only_prefix_probe(task):
 torch.set_num_threads(1);rng=np.random.default_rng(5);x=rng.normal(size=(24,5)).astype(np.float32);y=np.repeat([1,8],12) if task=='classification' else x[:,:1]**2
 model=TaskSyntheticLearner(5,task,epochs=1,device='cpu',variant='task');probe=DriftDiagnostic(5,task)
 for stage,ids in enumerate([slice(0,12),slice(12,24)],1):
  model.observe(x[ids],y[ids]);before=copy.deepcopy(model.rng.bit_generator.state);g=model.g.copy();params={k:v.clone() for k,v in model.function.net.state_dict().items()}
  probe.add(x[ids],y[ids],model,stage);rows=probe.evaluate(model,stage)
  assert model.rng.bit_generator.state==before;np.testing.assert_array_equal(g,model.g)
  for k,v in params.items():torch.testing.assert_close(v,model.function.net.state_dict()[k],rtol=0,atol=0)
  assert sum(r['n'] for r in rows)==12*stage and max(r['birth'] for r in rows)==stage

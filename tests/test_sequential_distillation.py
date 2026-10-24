import numpy as np
import torch
from tmot.warm_functional_basis import WarmFunctionalBasis,distillation_error
from tmot.replay_functional import IdentityBasis

def test_old_logits_only_and_teacher_detached():
 teacher=torch.tensor([[1.,2.]],requires_grad=True)
 student=torch.tensor([[1.,2.,100.]],requires_grad=True)
 loss=distillation_error(student,teacher,'classification').sum()
 assert abs(float(loss))<1e-6
 loss.backward();assert teacher.grad is None;assert student.grad[0,2]==0

def test_first_stage_control_and_teacher_immutable():
 for task in ('classification','regression'):
  x=np.random.default_rng(5).normal(size=(12,4)).astype('float32');y=np.arange(12)%2 if task=='classification' else x[:,0,None]
  def fit(strength,previous=None):return WarmFunctionalBasis(IdentityBasis(4),task,epochs=2,device='cpu').fit(x,y if previous is None or task=='regression' else y+2,previous=previous,distillation=strength,input_stats=None if previous is None else (previous.mean,previous.sd))
  a=fit(0);b=fit(1)
  for p,q in zip(a.net.parameters(),b.net.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
  before={k:v.clone() for k,v in a.net.state_dict().items()};c=fit(1,a)
  for k,v in before.items():torch.testing.assert_close(v,a.net.state_dict()[k],rtol=0,atol=0)
  assert c.warm_init_error==0 and np.isfinite(c.predict_head(x)).all()
  if task=='classification':assert c.classes==[0,1,2,3]

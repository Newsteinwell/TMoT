import numpy as np
import torch
from tmot.warm_task_budget import WarmER,WarmMomentER
from tmot.warm_functional_basis import WarmFunctionalBasis
from tmot.replay_functional import IdentityBasis

def test_warm_head_expansion_preserves_old_parameters():
 x=np.random.default_rng(1).normal(size=(12,4)).astype('float32')
 a=WarmFunctionalBasis(IdentityBasis(4),'classification',epochs=1,device='cpu').fit(x,np.arange(12)%2+7)
 b=WarmFunctionalBasis(IdentityBasis(4),'classification',epochs=0,device='cpu').fit(x,np.arange(12)%2,previous=a,input_stats=(a.mean,a.sd))
 assert b.classes==[7,8,0,1]
 for p,q in zip(a.net.parameters(),b.net.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
 torch.testing.assert_close(a.head.weight,b.head.weight[:2],rtol=0,atol=0)
 torch.testing.assert_close(a.head.bias,b.head.bias[:2],rtol=0,atol=0)

def test_warm_stream_zero_quota_and_regression():
 for cls in (WarmER,WarmMomentER):
  for task in ('classification','regression'):
   m=cls(4,task=task,mode='er',epochs=1,device='cpu')
   for i,n in enumerate((8,11,23)):
    x=np.random.default_rng(n).normal(size=(n,4)).astype('float32');y=np.arange(n)%2+i*2 if task=='classification' else x[:,0]
    m.observe(x,y)
    assert m.function.warm_init_error==0
    assert len(m.buffer[0])==sum(a['quota'] for a in m.buffer_audit)
    assert np.isfinite(m.function.predict_head(x)).all()
   if task=='classification':assert m.function.classes==m.classes==[0,1,2,3,4,5]

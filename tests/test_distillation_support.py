import numpy as np
from tmot.distillation_support import DistillationSupport

def test_arrival_only_and_real_quota():
 x=np.arange(32,dtype=np.float32).reshape(8,4);y=np.arange(8)%2
 m=DistillationSupport(4,'classification','er10');_,a=m.sample(x,y,16);assert not a['historical_support']
 m.update(x,y);assert m.audit['quota']==0
 _,a=m.sample(x+100,y+2,16);assert not a['historical_support']
 xx=np.tile(x,(3,1))+100;yy=np.tile(y,3)+2;m.update(xx,yy);assert m.audit['quota']==2 and m.audit['after']==2
 hx,a=m.sample(x+1000,y+4,16);assert a['historical_support'];assert all(any(np.array_equal(v,w) for w in xx) for v in hx)

def test_local_support_deterministic_and_prearrival():
 x=np.random.default_rng(3).normal(size=(24,4)).astype('float32');y=np.arange(24)%2
 a=DistillationSupport(4,'classification','local');b=DistillationSupport(4,'classification','local')
 for m in (a,b):m.update(x,y)
 ha,aa=a.sample(x+100,y+2,32);hb,ab=b.sample(x+100,y+2,32)
 np.testing.assert_array_equal(ha,hb);assert aa==ab and aa['historical_support'];assert abs(ha.mean())<2

def test_support_placeholder_has_no_supervised_gradient():
 import torch
 from tmot.warm_functional_basis import WarmFunctionalBasis
 from tmot.replay_functional import IdentityBasis
 x=np.random.default_rng(4).normal(size=(12,4)).astype('float32');y=x[:,0,None].copy();other=y.copy();other[4:]=999
 sw=np.r_[np.ones(4),np.zeros(8)];p=np.zeros((12,1),dtype='float32')
 a=WarmFunctionalBasis(IdentityBasis(4),'regression',device='cpu',epochs=2).fit(x,y,sample_weight=np.ones(12),supervised_weight=sw,external_targets=p,distillation=10)
 b=WarmFunctionalBasis(IdentityBasis(4),'regression',device='cpu',epochs=2).fit(x,other,sample_weight=np.ones(12),supervised_weight=sw,external_targets=p,distillation=10)
 for m,n in zip(a.net.parameters(),b.net.parameters()):torch.testing.assert_close(m,n,rtol=0,atol=0)
 torch.testing.assert_close(a.head.weight,b.head.weight,rtol=0,atol=0)

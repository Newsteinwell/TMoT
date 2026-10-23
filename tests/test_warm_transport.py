import numpy as np
import torch
from torch import nn
from tmot.nonlinear_moments import transport_residual_network,NonlinearMomentLearner

def test_affine_normalization_transport_and_class_expansion():
    torch.manual_seed(4)
    old=nn.Sequential(nn.Linear(5,8),nn.ReLU(),nn.Linear(8,8),nn.ReLU(),nn.Linear(8,3))
    new=nn.Sequential(nn.Linear(5,8),nn.ReLU(),nn.Linear(8,8),nn.ReLU(),nn.Linear(8,5))
    rng=np.random.default_rng(1);x=rng.normal(size=(17,5)).astype('float32')
    om=rng.normal(size=5).astype('float32');nm=om+2
    os=np.linspace(.3,2,5,dtype='float32');ns=os*3
    transport_residual_network(old,new,om,os,nm,ns)
    with torch.no_grad():
        a=old(torch.tensor((x-om)/os));b=new(torch.tensor((x-nm)/ns))
    torch.testing.assert_close(a,b[:,:3],atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(b[:,3:],torch.zeros_like(b[:,3:]),atol=0,rtol=0)

def test_warm_and_cold_first_stage_and_update_counts():
    torch.set_num_threads(1);rng=np.random.default_rng(12)
    x=rng.normal(size=(30,5)).astype('float32');y=np.repeat([3,9,15],10)
    a=NonlinearMomentLearner(5,epochs=2,width=8,device='cpu',replay_count=16,warm_start=True)
    b=NonlinearMomentLearner(5,epochs=2,width=8,device='cpu',replay_count=16)
    for m in (a,b):m.observe(x[:20],y[:20])
    for p,q in zip(a.net.parameters(),b.net.parameters()):torch.testing.assert_close(p,q,atol=0,rtol=0)
    for m in (a,b):m.observe(x[20:]+2,y[20:])
    assert a.classes==[3,9,15] and a.warm_init_error<1e-5
    assert a.optimizer_steps==b.optimizer_steps and a.historical_training_samples==b.historical_training_samples
    np.testing.assert_array_equal(a.base,b.base)

def test_warm_vector_regression():
    torch.set_num_threads(1);rng=np.random.default_rng(2)
    m=NonlinearMomentLearner(4,task='regression',epochs=1,width=8,device='cpu',replay_count=16,warm_start=True)
    for _ in range(2):m.observe(rng.normal(size=(12,4)),rng.normal(size=(12,2)))
    assert m.predict(np.zeros((3,4))).shape==(3,2)
    assert m.warm_init_error<1e-5

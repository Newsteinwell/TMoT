import numpy as np
import pytest
import torch
from tmot.local_moments import LocalCell,LocalMomentMixture
from tmot.latent_replay import LatentCell,LatentMomentMixture,LatentReplayLearner
from tmot.regularized_innovation import RegularizedInnovationLearner,regularized_innovation_readout


def test_equal_stored_moments_bounded_support_and_matched_rng():
    rng=np.random.default_rng(71);x=rng.normal(size=(50,5));x[:,4]=x[:,0]+.3*x[:,1]
    c=LocalCell.fit(x,rank=3);gauss=LatentCell(c.n,c.mean,c.factor,c.residual_sd,'gaussian');bounded=LatentCell(c.n,c.mean,c.factor,c.residual_sd,'rademacher')
    r1=np.random.default_rng(5);r2=np.random.default_rng(5);r3=np.random.default_rng(5)
    np.testing.assert_array_equal(c.sample(100,r1),gauss.sample(100,r2));bounded.sample(100,r3)
    assert r1.random()==r2.random()==r3.random()
    expected=c.factor.astype(float)@c.factor.astype(float).T+np.diag(c.residual_sd.astype(float)**2)
    for cell in (gauss,bounded):
        draws=cell.sample(100000,np.random.default_rng(8)).astype(float)
        np.testing.assert_allclose(draws.mean(0),c.mean,atol=.015)
        np.testing.assert_allclose(np.cov(draws,rowvar=False,bias=True),expected,atol=.02,rtol=.02)
    draws=bounded.sample(1000,np.random.default_rng(3));bound=np.abs(c.factor).sum(1)+c.residual_sd
    assert np.all(abs(draws-c.mean)<=bound+1e-5)

@pytest.mark.parametrize('task',['classification','regression'])
def test_same_component_fit_and_class_or_target_sampling(task):
    rng=np.random.default_rng(4);x=rng.normal(size=(40,4));y=np.repeat([2,7],20) if task=='classification' else np.c_[x[:,0]*2,x[:,0]*-3]
    kwargs=dict(dimension=4,task=task,seed=8,preserve_targets=True,support=4,min_cell=4)
    old=LocalMomentMixture(**kwargs);g=LatentMomentMixture(**kwargs);r=LatentMomentMixture(**kwargs,law='rademacher')
    for m in (old,g,r):m.update(x,y)
    classes=[2,7] if task=='classification' else []
    a,b=old.sample(100,np.array(classes).tolist(),np.random.default_rng(9));c,d=g.sample(100,classes,np.random.default_rng(9))
    np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
    for label in old.cells:
        for left,right in zip(old.cells[label],r.cells[label]):
            for field in ('mean','factor','residual_sd'):np.testing.assert_array_equal(getattr(left,field),getattr(right,field))
    if task=='regression':
        a,b=r.sample(10000,classes,np.random.default_rng(9));np.testing.assert_allclose(b[:,1],-1.5*b[:,0],atol=2e-6)

@pytest.mark.parametrize('task',['classification','regression'])
@pytest.mark.parametrize('mode',['local','er'])
def test_shared_readout_grid_equals_independent_training(task,mode):
    torch.set_num_threads(1);rng=np.random.default_rng(8);x=rng.normal(size=(24,4)).astype(np.float32)
    y=np.repeat([2,7],12) if task=='classification' else x[:,:1]**2
    kwargs=dict(dimension=4,task=task,mode=mode,epochs=1,device='cpu',replay_count=32)
    models={r:RegularizedInnovationLearner(**kwargs,ridge=r) for r in (.001,.01,.1,1.,10.,100.)};shared=LatentReplayLearner(**kwargs,ridge=.1)
    for sl in (slice(0,12),slice(12,24)):
        shared.observe(x[sl],y[sl])
        for ridge,model in models.items():
            model.observe(x[sl],y[sl]);weights,_=regularized_innovation_readout(shared.g,shared.k,4,ridge)
            np.testing.assert_array_equal(model.g,shared.g);np.testing.assert_array_equal(model.k,shared.k)
            np.testing.assert_array_equal(model.readout,weights)

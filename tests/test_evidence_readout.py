import numpy as np
import torch
from tmot.evidence_readout import evidence_objective,EvidenceReadout


def test_moment_evidence_matches_dense_observation_covariance():
    rng=np.random.default_rng(9);x=rng.normal(size=(13,4));y=rng.normal(size=(13,2))
    x-=x.mean(0);y-=y.mean(0)
    # Orthogonal basis removes the fitted intercept from observation space.
    q=np.linalg.qr(np.column_stack([np.ones(13),rng.normal(size=(13,12))]))[0][:,1:]
    x=q.T@x;y=q.T@y
    alpha=np.array([.3,.8,1.2,2.]);beta=1.7
    args=[torch.tensor(v,dtype=torch.float64) for v in (x.T@x,x.T@y,np.sum(y*y),np.log(alpha),np.log(beta))]
    value,mean=evidence_objective(*args[:3],12,*args[3:])
    cov=np.eye(12)/beta+(x/alpha)@x.T
    expected=.5*(2*np.linalg.slogdet(cov)[1]+np.sum(y*np.linalg.solve(cov,y)))
    np.testing.assert_allclose(value.item(),expected,rtol=1e-10)
    np.testing.assert_allclose(mean.numpy(),np.linalg.solve(np.diag(alpha)+beta*x.T@x,beta*x.T@y),rtol=1e-10)


def test_evidence_gradient_and_class_growth():
    rng=np.random.default_rng(3);x=rng.normal(size=(20,5));y=np.repeat([2,7,11,19],5)
    model=EvidenceReadout(5,mode='ard',steps=3,device='cpu')
    model.observe(x[:10],y[:10]);model.observe(x[10:],y[10:])
    assert set(model.predict(x))<=set(y)
    assert model.diagnostics['best_evidence']<=model.diagnostics['initial_evidence']
    tx=torch.tensor(x,dtype=torch.float64);ty=torch.tensor(rng.normal(size=(20,2)),dtype=torch.float64)
    la=torch.zeros(5,dtype=torch.float64,requires_grad=True);lb=torch.zeros((),dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(lambda a,b:evidence_objective(tx.T@tx,tx.T@ty,(ty*ty).sum(),20,a,b)[0],(la,lb))


def test_streaming_regression_matches_joint_moments():
    rng=np.random.default_rng(4);x=rng.normal(size=(25,6));y=x[:,:2]+.1*rng.normal(size=(25,2))
    a=EvidenceReadout(6,task='regression',steps=3,device='cpu');b=EvidenceReadout(6,task='regression',steps=3,device='cpu')
    a.observe(x[:12],y[:12]);a.observe(x[12:],y[12:]);b.observe(x,y)
    np.testing.assert_allclose(a.predict(x),b.predict(x),atol=1e-10)


def test_balanced_class_statistics_match_weighted_observations():
    from tmot.moments import ClassificationMoments
    rng=np.random.default_rng(6);x=rng.normal(size=(16,4));labels=np.array([0]*4+[1]*12);y=np.eye(2)[labels]
    memory=ClassificationMoments(4);memory.update(x,labels);g,k,v=memory.risk_moments([0,1])
    w=np.array([16/(2*np.sum(labels==c)) for c in labels]);mx=np.average(x,axis=0,weights=w);my=np.average(y,axis=0,weights=w)
    dx=x-mx;dy=y-my
    np.testing.assert_allclose(16*(g[:-1,:-1]-np.outer(g[:-1,-1],g[:-1,-1])),dx.T@(w[:,None]*dx),atol=1e-12)
    np.testing.assert_allclose(16*(k[:-1]-np.outer(g[:-1,-1],k[-1])),dx.T@(w[:,None]*dy),atol=1e-12)
    np.testing.assert_allclose(16*(v-k[-1]@k[-1]),np.sum(w[:,None]*dy*dy),atol=1e-12)

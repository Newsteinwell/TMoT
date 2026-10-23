import numpy as np
import torch
from tmot.moments import (ClassificationMoments,RegressionMoments,augment,affine_matrix,
                          transport_risk,quadratic_risk,ridge_readout)


def test_growing_classes_loss_and_gradient():
    rng=np.random.default_rng(8);x=rng.normal(size=(21,5));y=np.array([3]*7+[11]*14)
    m=ClassificationMoments(5);m.update(x,y)
    classes=[3,11,20,8]  # new output rows exist although no old samples have these labels
    g,k,v=m.risk_moments(classes)
    f=torch.tensor(rng.normal(size=(4,6)),requires_grad=True)
    closed=quadratic_risk(f,torch.tensor(g),torch.tensor(k),torch.tensor(v))
    target=np.eye(4)[[classes.index(c) for c in y]]
    err=((torch.tensor(augment(x))@f.T-torch.tensor(target))**2).sum(1)
    direct=(err[:7].mean()+err[7:].mean())/2
    np.testing.assert_allclose(closed.detach(),direct.detach(),rtol=1e-12)
    gc=torch.autograd.grad(closed,f,retain_graph=True)[0]
    gd=torch.autograd.grad(direct,f)[0]
    torch.testing.assert_close(gc,gd,rtol=1e-12,atol=1e-12)


def test_regression_vector_exact_merge_and_transport():
    rng=np.random.default_rng(17);x=rng.normal(size=(50,7));y=rng.normal(size=(50,3))
    m=RegressionMoments(7,3);m.update(x[:17],y[:17]);m.update(x[17:],y[17:])
    g,k,v=m.risk_moments();a=rng.normal(size=(4,7));b=rng.normal(size=4);h=affine_matrix(a,b)
    gh,kh=transport_risk(g,k,h);hx=augment(x@a.T+b)
    np.testing.assert_allclose(gh,hx.T@hx/len(x),atol=1e-12)
    np.testing.assert_allclose(kh,hx.T@y/len(x),atol=1e-12)
    f=torch.tensor(rng.normal(size=(3,5)),requires_grad=True)
    c=quadratic_risk(f,torch.tensor(gh),torch.tensor(kh),torch.tensor(v))
    direct=((torch.tensor(hx)@f.T-torch.tensor(y))**2).sum(1).mean()
    torch.testing.assert_close(c,direct,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(c,f,retain_graph=True)[0],torch.autograd.grad(direct,f)[0])
    r=ridge_readout(g,k,.1)
    rr=np.linalg.solve(augment(x).T@augment(x)/len(x)+np.diag([.1]*7+[0]),augment(x).T@y/len(x))
    np.testing.assert_allclose(r,rr,atol=1e-12)


def test_class_merge_includes_between_mean_scatter():
    x=np.array([[0.,1.],[2.,3.],[10.,12.],[20.,30.]])
    m=ClassificationMoments(2);m.update(x[:2],np.array([7,7]));m.update(x[2:],np.array([7,7]))
    g,k,_=m.risk_moments([7,13]);z=augment(x)
    np.testing.assert_allclose(g,z.T@z/4)
    np.testing.assert_allclose(k[:,0],z.mean(0));assert np.all(k[:,1]==0)


def test_dynamic_head_and_checkpoint_have_no_replay():
    from tmot.learner import MomentLearner
    rng=np.random.default_rng(4);x=rng.normal(size=(24,8)).astype('float32')
    m=MomentLearner(8,epochs=1,bottleneck=4)
    m.observe(x[:12],np.array([0]*6+[9]*6));m.observe(x[12:],np.array([21]*6+[9]*6))
    assert m.classes==[0,9,21];assert m.buffer is None
    assert m.predict(x).shape==(24,);assert set(m.predict(x)).issubset({0,9,21})


def test_regression_learner_multidimensional():
    from tmot.learner import MomentLearner
    rng=np.random.default_rng(5);x=rng.normal(size=(30,8)).astype('float32');y=x[:,:2]*2
    m=MomentLearner(8,task='regression',epochs=1,bottleneck=4)
    m.observe(x[:15],y[:15]);m.observe(x[15:],y[15:])
    assert m.predict(x).shape==(30,2);assert np.isfinite(m.predict(x)).all();assert m.buffer is None

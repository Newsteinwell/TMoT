import numpy as np
from tmot.moments import Cell, augment, ridge_readout
from tmot.nonlinear_moments import sample_cell, NonlinearMomentLearner


def test_gaussian_replay_joint_mean_and_shrunk_covariance():
    rng=np.random.default_rng(9);x=rng.normal(size=(80,3));y=x[:,0:1]*2+x[:,1:2]
    joint=np.c_[x,y];cell=Cell.from_x(joint)
    generated=sample_cell(cell,80000,rng)
    cov=np.cov(joint,rowvar=False,bias=True);expected=.9*cov+.1*np.diag(np.diag(cov))
    np.testing.assert_allclose(generated.mean(0),joint.mean(0),atol=.025)
    np.testing.assert_allclose(np.cov(generated,rowvar=False,bias=True),expected,atol=.06)


def test_zero_residual_equals_cumulative_ridge_and_no_samples_retained():
    rng=np.random.default_rng(3);x=rng.normal(size=(30,4)).astype('float32');y=rng.normal(size=(30,2))
    m=NonlinearMomentLearner(4,'regression',device='cpu',epochs=1,replay_count=32,width=8,alpha=0.)
    m.observe(x[:15],y[:15]);m.observe(x[15:],y[15:])
    g,k,_=m.memory.risk_moments();expected=augment(x)@ridge_readout(g,k,.1)
    np.testing.assert_allclose(m.predict(x),expected,atol=1e-8)
    assert m.buffer is None and len(m.joint_cells)==2 and m.seen==30


def test_class_growth_replay_only_arrived_classes_and_er_budget():
    rng=np.random.default_rng(5);x=rng.normal(size=(24,4)).astype('float32');y=np.repeat([9,2,7],8)
    for mode in ('gaussian','er'):
        m=NonlinearMomentLearner(4,device='cpu',epochs=1,replay_count=32,width=8,mode=mode,replay_bytes=256)
        m.observe(x[:16],y[:16]);hx,hy=m._historical(None)
        assert set(hy)<=set([2,9])
        m.observe(x[16:],y[16:]);assert set(m.classes)==set([2,7,9])
        assert m.predict(x).shape==(24,)
        if mode=='er':assert sum(v.nbytes for v in m.buffer)+m.keys.nbytes<=256


def test_matched_replay_uses_identical_update_counts():
    rng=np.random.default_rng(4);x=rng.normal(size=(20,4)).astype('float32');y=np.repeat([2,7],10)
    models=[]
    for mode in ('gaussian','er'):
        model=NonlinearMomentLearner(4,mode=mode,device='cpu',epochs=2,width=8,
                                    replay_count=300,replay_bytes=256,er_resample=True)
        model.observe(x[:10],y[:10]);model.observe(x[10:],y[10:]);models.append(model)
    assert [m.historical_training_samples for m in models]==[300,300]
    assert [m.optimizer_steps for m in models]==[4,4]


def test_task_losses_have_expected_invariances_and_robust_tail():
    import torch
    from tmot.nonlinear_moments import per_sample_loss
    scores=torch.tensor([[2.,-.5,1.],[-1.,3.,.2]],requires_grad=True);target=torch.eye(3)[torch.tensor([0,1])]
    loss=per_sample_loss(scores,target,'classification','task')
    torch.testing.assert_close(loss,per_sample_loss(scores+100,target,'classification','task'))
    loss.sum().backward();assert torch.isfinite(scores.grad).all()
    small=torch.tensor([[.5,-.8]])
    torch.testing.assert_close(per_sample_loss(small,torch.zeros_like(small),'regression','task'),per_sample_loss(small,torch.zeros_like(small),'regression','mse'))
    tail=torch.tensor([[10.]],requires_grad=True)
    per_sample_loss(tail,torch.zeros_like(tail),'regression','task').sum().backward()
    assert tail.grad.item()==2.


def test_task_loss_supports_class_growth_and_vector_regression():
    rng=np.random.default_rng(8);x=rng.normal(size=(24,4)).astype('float32')
    for task,y in [('classification',np.repeat([2,7,9],8)),('regression',rng.normal(size=(24,2)))]:
        m=NonlinearMomentLearner(4,task=task,loss='task',device='cpu',epochs=1,width=8,replay_count=32)
        m.observe(x[:16],y[:16]);m.observe(x[16:],y[16:])
        assert np.isfinite(m.predict(x)).all()
        assert m.predict(x).shape==y.shape

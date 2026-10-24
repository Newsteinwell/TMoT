import numpy as np
import pytest
import torch
from tmot.local_moments import LocalMomentMixture
from tmot.covariance_replay import CovarianceMixture,CovarianceReplayLearner,VARIANTS
from tmot.regularized_innovation import RegularizedInnovationLearner

@pytest.mark.parametrize('task',['classification','regression'])
def test_full_covariance_and_empirical_support(task):
    rng=np.random.default_rng(4);x=rng.normal(size=(20,30));x[:,2]=x[:,0]+x[:,1];y=np.zeros(20,dtype=int) if task=='classification' else np.c_[x[:,3]**2,2*x[:,3]**2]
    mixture=CovarianceMixture(30,task,preserve_targets=True,components=1,covariance='full_shrink0');mixture.update(x,y)
    cell=next(iter(mixture.cells.values()))[0];rows=x if task=='classification' else np.c_[x,y];center=rows-rows.mean(0)
    cov=cell.factor.astype(float)@cell.factor.astype(float).T+np.diag(cell.residual_sd.astype(float)**2)
    np.testing.assert_allclose(cov,center.T@center/len(rows),rtol=1e-5,atol=2e-6)
    _,v,q=np.linalg.svd(center,full_matrices=False);q=q[v>v.max()*1e-10]
    draws=cell.sample(200,np.random.default_rng(3)).astype(float)-rows.mean(0)
    np.testing.assert_allclose(draws,draws@q.T@q,atol=2e-6,rtol=1e-5)
    assert not np.any(cell.residual_sd)

@pytest.mark.parametrize('task',['classification','regression'])
def test_same_allocations_and_legacy_control(task):
    rng=np.random.default_rng(7);x=rng.normal(size=(200,40));y=np.repeat([2,8],100) if task=='classification' else x[:,:1]**2
    kwargs=dict(dimension=40,task=task,seed=7,preserve_targets=True,support=4,min_cell=4)
    old=LocalMomentMixture(**kwargs);models=[CovarianceMixture(**kwargs,covariance=v) for v in VARIANTS];streams=[np.random.default_rng(9) for _ in models]
    old.update(x,y)
    for m in models:m.update(x,y)
    for label,cells in old.cells.items():
        for index,base in enumerate(cells):
            expected=base.factor.astype(float)@base.factor.astype(float).T+np.diag(base.residual_sd.astype(float)**2)
            for model in models:
                cell=model.cells[label][index];actual=cell.factor.astype(float)@cell.factor.astype(float).T+np.diag(cell.residual_sd.astype(float)**2)
                np.testing.assert_array_equal(cell.mean,base.mean);assert cell.n==base.n
                np.testing.assert_allclose(np.diag(actual),np.diag(expected),rtol=1e-5,atol=2e-6)
                if task=='regression':np.testing.assert_allclose(actual[-1],expected[-1],rtol=1e-5,atol=2e-6)
    labels=[2,8] if task=='classification' else [];ox,oy=old.sample(513,labels,np.random.default_rng(9))
    assert any(c.factor.shape[1]>c.reference_width for cells in models[-1].cells.values() for c in cells)
    samples=[m.sample(513,labels,r) for m,r in zip(models,streams)]
    np.testing.assert_array_equal(samples[0][0],ox);np.testing.assert_array_equal(samples[0][1],oy)
    assert all(r.bit_generator.state==streams[0].bit_generator.state for r in streams)
    if task=='classification':
        for _,labels_new in samples:np.testing.assert_array_equal(labels_new,oy)

@pytest.mark.parametrize('task',['classification','regression'])
def test_control_training_unchanged(task):
    torch.set_num_threads(1);rng=np.random.default_rng(1);x=rng.normal(size=(24,5));y=np.repeat([1,9],12) if task=='classification' else x[:,:1]
    kwargs=dict(dimension=5,task=task,epochs=1,device='cpu',replay_count=32)
    old=RegularizedInnovationLearner(**kwargs);new=CovarianceReplayLearner(**kwargs)
    for sl in (slice(0,12),slice(12,24)):
        old.observe(x[sl],y[sl]);new.observe(x[sl],y[sl]);np.testing.assert_array_equal(old.readout,new.readout)

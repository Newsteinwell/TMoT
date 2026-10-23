import numpy as np
import pytest
from tmot.dual_basis import DualPCABasis
from tmot.transport_learner import TransportLearner

@pytest.mark.parametrize('scenario',['cil','dil','regression'])
def test_generalized_transport_and_predictions(scenario):
    rng=np.random.default_rng(11);x=rng.normal(size=(16,18)).astype('float32')
    y=np.repeat([0,1],8);x[:,0]+=y*3
    task='regression' if scenario=='regression' else 'classification'
    if task=='regression':y=np.column_stack([x[:,0],x[:,1]])
    basis=DualPCABasis(2,task,epochs=1,device='cpu').fit(x,y)
    learner=TransportLearner(basis,epochs=1)
    learner.observe(basis.transform(x),y)
    nx=x+.1;ny=y+2 if scenario=='cil' else y
    learner.observe(basis.transform(nx),ny)
    pred=learner.predict(basis.transform(nx));assert np.isfinite(pred).all()
    if scenario=='cil':
        assert learner.classes==[0,1,2,3]
        assert learner.calibration['support']=='pooled_observed_classes'
        assert set(learner.domains[0].cells)=={0,1} and set(learner.domains[1].cells)=={2,3}
    elif scenario=='dil':assert learner.calibration['support']=='domain_experts'
    else:assert pred.shape==(16,2)

def test_growing_centered_history_equals_explicit_old_rows():
    import torch
    from tmot.moments import ClassificationMoments,quadratic_risk
    rng=np.random.default_rng(31);x=rng.normal(size=(10,7));y=np.array([0]*4+[1]*6)
    memory=ClassificationMoments(7);memory.update(x,y)
    learner=object.__new__(TransportLearner);learner.classes=[0,1,2,3];learner.task='classification';learner.device='cpu'
    moments=learner._risk(memory)
    f=torch.tensor(rng.normal(size=(4,8)),dtype=torch.float64);f=f-f.mean(0,keepdim=True)
    targets=(np.eye(4)-np.ones((4,4))/4)/np.sqrt(.75)
    pred=np.column_stack([x,np.ones(len(x))])@f.numpy().T
    explicit=np.mean([np.square(pred[y==c]-targets[c]).sum(1).mean() for c in (0,1)])
    np.testing.assert_allclose(quadratic_risk(f,*moments).item(),explicit,rtol=1e-12)


def test_transport_statistics_equal_direct_affine_data():
    import torch
    from tmot.moments import ClassificationMoments
    from tmot.transport_learner import transported
    from tmot.wimot_core.statistics import moments
    rng=np.random.default_rng(32);x=rng.normal(size=(20,7));y=np.repeat([0,1],10)
    memory=ClassificationMoments(7);memory.update(x,y)
    adapter=torch.nn.Linear(7,3).double()
    estimate=transported(memory,[0,1],adapter)
    h=x@adapter.weight.detach().numpy().T+adapter.bias.detach().numpy()
    direct=moments(h,y,2)
    np.testing.assert_allclose(estimate.means,direct.means,atol=1e-12)
    np.testing.assert_allclose(estimate.covariances,direct.covariances,atol=1e-12)

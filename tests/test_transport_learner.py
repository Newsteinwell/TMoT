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

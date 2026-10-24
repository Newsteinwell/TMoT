import numpy as np
from tmot.strict_residual_replay import StrictResidualER
from tmot.learner import MomentLearner
import torch

def test_empty_quotas_and_exact_analytic_base():
    m=StrictResidualER(4,task='regression',mode='er',epochs=1,device='cpu',warm_start=True,replay_count=16)
    b=MomentLearner(4,'regression','ridge',ridge=.1)
    expected=0
    for n in (5,21,30):
        rng=np.random.default_rng(n);x=rng.normal(size=(n,4)).astype('float32');y=x[:,:1]+.1*rng.normal(size=(n,1))
        m.observe(x,y);b.observe(x,y)
        assert m.historical_training_samples==(16 if expected else 0)
        expected+=n//10
        assert len(m.buffer[0])==expected
        assert m.buffer_audit[-1]['quota']==n//10
        np.testing.assert_array_equal(m.base,b.readout)
        assert np.isfinite(m.predict(x)).all()
        if m.stage>1:assert m.warm_init_error<1e-5

def test_cold_restart_discards_previous_network_but_keeps_memory():
    models=[StrictResidualER(4,task='regression',mode='er',epochs=1,device='cpu',warm_start=False,replay_count=16) for _ in range(2)]
    rng=np.random.default_rng(55)
    x=rng.normal(size=(20,4)).astype('float32');y=x[:,:1]
    for m in models:m.observe(x,y)
    with torch.no_grad():models[1].net[-1].bias.add_(100)
    z=rng.normal(size=(20,4)).astype('float32')
    for m in models:m.observe(z,z[:,:1])
    np.testing.assert_array_equal(models[0].predict(x),models[1].predict(x))
    assert models[0].seen==40 and models[0].historical_training_samples==16
    assert models[0].buffer_audit==models[1].buffer_audit

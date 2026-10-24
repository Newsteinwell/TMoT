import numpy as np
from tmot.strict_residual_replay import StrictResidualER
from tmot.learner import MomentLearner

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

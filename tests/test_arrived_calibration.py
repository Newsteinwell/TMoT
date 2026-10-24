import numpy as np
import pytest
import torch
from tmot.arrived_calibration import ArrivedCalibrationLearner
from tmot.regularized_innovation import RegularizedInnovationLearner

@pytest.mark.parametrize('task',['classification','regression'])
@pytest.mark.parametrize('mode',['local','er'])
def test_causal_arrived_statistics_and_unchanged_control(task,mode):
    torch.set_num_threads(1);rng=np.random.default_rng(119)
    x=np.r_[rng.normal(size=(16,4)),5+3*rng.normal(size=(20,4))].astype(np.float32)
    y=np.r_[np.zeros(16,dtype=int),np.ones(20,dtype=int)] if task=='classification' else x[:,:1]**2
    kwargs=dict(dimension=4,task=task,mode=mode,epochs=1,device='cpu',replay_count=32)
    original=RegularizedInnovationLearner(**kwargs)
    first=ArrivedCalibrationLearner(**kwargs,calibration='first')
    arrived=ArrivedCalibrationLearner(**kwargs,calibration='arrived')
    for stop,start in ((16,0),(36,16)):
        for model in (original,first,arrived):model.observe(x[start:stop],y[start:stop])
        np.testing.assert_array_equal(first.readout,original.readout)
        np.testing.assert_array_equal(first.predict(x[:stop]),original.predict(x[:stop]))
        assert arrived.calibration_n==stop
        np.testing.assert_allclose(arrived.calibration_mean,x[:stop].astype(float).mean(0),atol=1e-12)
        np.testing.assert_allclose(arrived.calibration_m2,((x[:stop].astype(float)-x[:stop].astype(float).mean(0))**2).sum(0),atol=1e-10)
        if stop==16:np.testing.assert_array_equal(arrived.readout,first.readout)
        else:
            np.testing.assert_allclose(arrived.function.mean,x.astype(float).mean(0),rtol=1e-6)
            np.testing.assert_allclose(arrived.function.sd,x.astype(float).std(0),rtol=1e-6)
        assert arrived.optimizer_steps==first.optimizer_steps
        assert np.isfinite(arrived.readout).all()
        np.testing.assert_array_equal(arrived.exact_memory.risk_moments(arrived.classes)[0],first.exact_memory.risk_moments(first.classes)[0])

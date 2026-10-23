import numpy as np
import torch
from tmot.replay_functional import representative_weights,ReplayFunctionalLearner

def test_class_masses_include_recurrent_and_new_classes():
    current=np.array([2,2,5,5,5]);history=np.array([2,7,7,7])
    w=representative_weights(current,history,'classification',{2:8,7:12},20)
    labels=np.r_[current,history]
    for c in [2,5,7]:np.testing.assert_allclose(w[labels==c].sum(),1/3)
    np.testing.assert_allclose(w[:2].sum(),(2/10)/3)

def test_regression_history_mass_does_not_depend_on_draw_count():
    current=np.zeros((10,2));history=np.zeros((400,2))
    w=representative_weights(current,history,'regression',{},90)
    np.testing.assert_allclose([w[:10].sum(),w[10:].sum()],[.1,.9])

def test_first_stage_pair_and_streaming_shapes():
    torch.set_num_threads(1);rng=np.random.default_rng(4);x=rng.normal(size=(24,5));y=np.repeat([3,8],12)
    models=[ReplayFunctionalLearner(5,mode=mode,epochs=1,device='cpu',replay_count=32) for mode in ('local','er')]
    for m in models:m.observe(x,y)
    np.testing.assert_array_equal(models[0].predict(x),models[1].predict(x))
    original=models[0].input_stats[0].copy()
    for m in models:m.observe(x[:12]+1,np.full(12,11))
    assert models[0].classes==[3,8,11] and models[0].buffer is None
    assert models[0].optimizer_steps==models[1].optimizer_steps
    np.testing.assert_array_equal(original,models[0].input_stats[0])
    assert sum(c.n for cells in models[0].local.cells.values() for c in cells)==36
    m=ReplayFunctionalLearner(5,'regression',epochs=1,device='cpu',replay_count=32)
    m.observe(x,rng.normal(size=(24,2)));m.observe(x[:12],rng.normal(size=(12,2)))
    assert m.predict(x[:3]).shape==(3,2) and np.isfinite(m.predict(x)).all()

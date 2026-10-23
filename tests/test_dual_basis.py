import numpy as np
import pytest
from tmot.dual_basis import DualPCABasis

@pytest.mark.parametrize('task',['classification','regression'])
def test_first_stage_freeze_and_rank(task):
    rng=np.random.default_rng(5);x=rng.normal(size=(12,18)).astype('float32')
    y=np.repeat([0,1],6) if task=='classification' else rng.normal(size=(12,2)).astype('float32')
    basis=DualPCABasis(2,task,epochs=1,device='cpu').fit(x,y)
    original=basis.transform(x);state={k:v.clone() for k,v in basis.stem.state_dict().items()}
    future=basis.transform(x+20)
    assert original.shape==future.shape==(12,512)
    assert basis.metadata['pca_rank']<=11
    np.testing.assert_array_equal(original,basis.transform(x))
    assert all((v==state[k]).all() for k,v in basis.stem.state_dict().items())
    with pytest.raises(RuntimeError):basis.fit(x,y)

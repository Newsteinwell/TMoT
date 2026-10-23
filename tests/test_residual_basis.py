import numpy as np
from tmot.residual_basis import RankCompletedBasis

def test_rank_completion_uses_orthogonal_directions():
    rng=np.random.default_rng(13);x=rng.normal(size=(12,302)).astype('float32');y=np.repeat([0,1],6)
    b=RankCompletedBasis(2,epochs=0,device='cpu').fit(x,y);k=b.metadata['pca_rank']
    assert k<=11 and b.metadata['rank_completion_rows']==256-k
    np.testing.assert_allclose(b.pca[:k]@b.pca[k:].T,0,atol=1e-6)
    assert np.linalg.matrix_rank(b.pca,tol=1e-5)>k

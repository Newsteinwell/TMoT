"""Complete low-rank first-stage PCA with fixed semantic directions.

This new extension retains the learned WiMoT components but prevents first-stage
sample rank from removing all other semantic directions. No future data are used.
"""
import numpy as np
from .dual_basis import DualPCABasis

class RankCompletedBasis(DualPCABasis):
    def fit(self,x,y):
        super().fit(x,y)
        k=self.metadata['pca_rank'];rng=np.random.default_rng(20261024)
        residual=rng.normal(size=(256-k,self.semantic_dim))
        if k:residual-=(residual@self.pca[:k].T)@self.pca[:k]
        residual/=np.maximum(np.linalg.norm(residual,axis=1,keepdims=True),1e-12)
        self.pca[k:]=residual.astype(np.float32)
        self.metadata['rank_completion_rows']=256-k
        self.metadata['rank_completion_seed']=20261024
        return self

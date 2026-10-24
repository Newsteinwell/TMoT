import numpy as np
from scripts.diagnose_teacher_complementarity import residual_decomposition

def test_opposite_errors_mix_cancels():
 y=np.array([[0.],[2.]])
 d=residual_decomposition(y,y+1,y-1)
 assert d['residual_cosine']==-1 and d['fixed_mixtures']['0.5']==0
 assert d['label_oracle_beta']==.5 and d['label_oracle_rmse']==0

def test_identical_and_endpoint_oracle():
 y=np.array([0.,1.])
 d=residual_decomposition(y,y+1,y+1);assert d['label_oracle_beta']==0 and d['label_oracle_rmse']==1
 d=residual_decomposition(y,y,y+2);assert d['label_oracle_beta']==0 and d['fixed_mixtures']['0.25']==.5

import numpy as np
from scripts.diagnose_readout_complementarity import decomposition

def test_class_balance_and_error_partition():
 # Class 0 has two examples, class 1 only one; each class contributes half.
 result=decomposition(np.array([0,0,1]),np.array([0,1,1]),np.array([1,0,1]))
 assert result['rescue']==.25 and result['harm']==.25
 assert result['both_correct']==.5 and result['both_wrong']==0
 assert result['reference_accuracy']==result['candidate_accuracy']==.75
 assert result['label_oracle_selector_accuracy']==1

def test_shared_errors_cannot_be_fixed_by_selector():
 result=decomposition(np.array([0,1]),np.array([0,0]),np.array([0,2]))
 assert result['both_wrong']==.5 and result['disagree']==.5
 assert result['rescue']==result['harm']==0
 assert result['label_oracle_selector_accuracy']==.5

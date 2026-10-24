import numpy as np
from tmot.class_distance_completion import weighted_stats,complete_class,fast_precision
from tmot.weighted_class_distance import precision

def test_completion_preserves_base_and_is_psd():
 rng=np.random.default_rng(21);x=rng.normal(size=(80,9));mu,cov=weighted_stats(x,np.ones(80),80);z=rng.normal(size=(30,5));em,ec=weighted_stats(z,np.ones(30),30)
 out,c,diag=complete_class(mu,cov,em,ec)
 np.testing.assert_array_equal(out[:5],em);np.testing.assert_array_equal(c[:5,:5],ec)
 assert np.linalg.eigvalsh(c).min()>-1e-12
 # Without ridge shrinkage, unchanged input statistics reconstruct joint moments.
 m2,c2,_=complete_class(mu,cov,mu[:5],cov[:5,:5],rho=0.)
 np.testing.assert_allclose(m2,mu,atol=1e-12);np.testing.assert_allclose(c2,cov,atol=1e-12)

def test_cholesky_matches_fecam_inverse():
 x=np.random.default_rng(22).normal(size=(40,13));_,cov=weighted_stats(x,np.ones(40),40)
 for alpha in (.01,.1,.5,1,2,10):np.testing.assert_allclose(fast_precision(cov,alpha),precision(cov,alpha),rtol=1e-9,atol=1e-10)

def test_two_arrivals_keep_training_state_and_current_only_unit_memory():
 import pickle
 from tmot.task_synthetic_replay import TaskSyntheticLearner
 from tmot.moments import ClassificationMoments
 from tmot.fecam import unit_rows
 from tmot.class_distance_completion import statistics,block_features,predict_stats
 m=TaskSyntheticLearner(4,'classification',42,mode='local',epochs=1,device='cpu',variant='task',points_per_cell=8);memory=ClassificationMoments(4)
 for stage in range(2):
  x=np.random.default_rng(stage).normal(size=(12,4)).astype('float32');y=np.arange(12)%2+stage*2;m.observe(x,y);memory.update(unit_rows(x),y)
  before=pickle.dumps(m.rng.bit_generator.state);points=[c.points.copy() for cells in m.local.cells.values() for c in cells];ss,_=statistics(m,memory)
  assert before==pickle.dumps(m.rng.bit_generator.state)
  for p,c in zip(points,[c for cells in m.local.cells.values() for c in cells]):np.testing.assert_array_equal(p,c.points)
  assert sum(c.n for c in memory.cells.values())==12*(stage+1)
  for mode in ss:
   features=unit_rows(x) if mode=='base' else block_features(m.function,x)
   assert np.isin(predict_stats(features,ss[mode],1.,m.classes),m.classes).all()

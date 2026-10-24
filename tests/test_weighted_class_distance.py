import numpy as np
from tmot.weighted_class_distance import class_statistics,precision,distances
from tmot.fecam import FeCAMLearner

def test_weighted_moments_equal_integer_replication():
 x=np.random.default_rng(8).normal(size=(9,5));w=np.arange(9)%3+1
 mu,cov,_=class_statistics(x,w,int(w.sum()));xx=np.repeat(x,w,axis=0);mm,cc,_=class_statistics(xx,np.ones(len(xx)),len(xx))
 np.testing.assert_allclose(mu,mm,atol=1e-15);np.testing.assert_allclose(cov,cc,atol=1e-15)

def test_unweighted_distance_matches_fecam():
 rng=np.random.default_rng(9);x=rng.normal(size=(27,7));y=np.arange(27)%3;test=rng.normal(size=(11,7));m=FeCAMLearner(7,.5);m.observe(x,y)
 for c in m.classes:
  xx=x[y==c];mu,cov,_=class_statistics(xx,np.ones(len(xx)),len(xx));p=precision(cov,.5)
  np.testing.assert_allclose(mu,m.means[c],atol=1e-15);np.testing.assert_allclose(p,m.precisions[c],atol=1e-12)
  np.testing.assert_allclose(distances(test,mu,p),m.distances(test)[:,m.classes.index(c)],atol=1e-12)

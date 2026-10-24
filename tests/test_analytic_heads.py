import numpy as np
from tmot.moments import ClassificationMoments
from tmot.analytic_heads import make_heads,predict_head

def test_cumulative_heads_equal_direct_arrived_training():
 rng=np.random.default_rng(52);x=rng.normal(size=(37,5));y=np.arange(37)%3;classes=[2,0,1];m=ClassificationMoments(5)
 for ids in np.array_split(np.arange(37),4):m.update(x[ids],y[ids])
 heads=make_heads(m,classes);mu=np.asarray([x[y==c].mean(0) for c in classes]);cov=sum((x[y==c]-u).T@(x[y==c]-u) for c,u in zip(classes,mu))/(len(x)-3)
 p=np.linalg.inv(.9*cov+.1*np.trace(cov)/5*np.eye(5));np.testing.assert_allclose(heads['slda_0.1']['precision'],p,atol=1e-12)
 test=rng.normal(size=(19,5));expected=np.asarray(classes)[((test[:,None,:]-mu[None,:,:])**2).sum(2).argmin(1)]
 np.testing.assert_array_equal(predict_head(heads['ncm'],test,classes),expected)
 z=np.c_[x,np.ones(len(x))];weights=np.asarray([1/(3*np.sum(y==c)) for c in y]);target=np.eye(3)[[classes.index(c) for c in y]]
 g=z.T@(weights[:,None]*z);k=z.T@(weights[:,None]*target);reg=np.eye(6)*.1;reg[-1,-1]=0
 np.testing.assert_allclose(heads['ridge_0.1']['weight'],np.linalg.solve(g+reg,k),atol=1e-12)

def test_singleton_slda_finite():
 m=ClassificationMoments(3);x=np.array([[0.,0.,0.],[1.,1.,1.]]);m.update(x,np.array([4,9]));heads=make_heads(m,[4,9])
 for h in heads.values():assert np.isfinite(predict_head(h,x,[4,9])).all()

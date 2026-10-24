import numpy as np
import torch
from tmot.certified_update import initial,head,certificate,scores,positive,step,regression_risk
from tmot.moments import ClassificationMoments,RegressionMoments

def test_spectral_positive_part():
 S=torch.diag(torch.tensor([2.,-3.,.5],dtype=torch.float64));H=positive(S)
 assert abs(float(torch.trace(H))-2.5)<1e-12
 assert float(torch.linalg.eigvalsh(H-S).min())>=-1e-12

def test_increment_bounds_and_identity():
 rng=np.random.default_rng(5);x=rng.normal(size=(60,8));y=np.repeat(np.arange(3),20)
 m=ClassificationMoments(8);m.update(x,y);a=head(initial(8,3,1),m,[0,1,2]);b=head(initial(8,3,2),m,[0,1,2])
 cert=certificate(a,b,m,True);delta=(scores(b,x)-scores(a,x)).numpy();emp=[]
 for c in range(3):emp.append(np.maximum(0,(delta[y==c]-delta[y==c,c,None]).max(1)).mean())
 assert np.mean(emp)<=cert['bound']+1e-9
 assert certificate(a,a,m,True)['bound']<1e-10

def test_exact_regression_risk():
 rng=np.random.default_rng(6);x=rng.normal(size=(40,8));y=rng.normal(size=(40,2));m=RegressionMoments(8,2);m.update(x,y)
 h=head(initial(8,3,1),m,[])
 np.testing.assert_allclose(regression_risk(h,m),np.mean(np.sum((scores(h,x).numpy()-y)**2,axis=1)),atol=1e-12)

def test_budget_acceptance_and_no_raw_history():
 rng=np.random.default_rng(7);x=rng.normal(size=(40,8));y=np.repeat([0,1],20);old=ClassificationMoments(8);old.update(x,y)
 m=ClassificationMoments(8);m.update(x,y);xn=rng.normal(size=(40,8))+2;yn=np.repeat([2,3],20);m.update(xn,yn)
 A,h,log,ref=step(initial(8,3,1),m,old,[0,1,2,3],xn,yn,'center',.01)
 if log['accepted']:
  row=log['proposals'][-1];assert row['certificate']['bound']<=.01 and row['gain']>=log['minimum_gain']
 np.testing.assert_allclose((A@A.T).numpy(),np.eye(3),atol=1e-12)

def test_bound_with_singular_historical_covariance():
 rng=np.random.default_rng(9);x=rng.normal(size=(6,12));y=np.repeat([0,1,2],2);m=ClassificationMoments(12);m.update(x,y)
 a=head(initial(12,4,1),m,[0,1,2]);b=head(initial(12,4,2),m,[0,1,2]);c=certificate(a,b,m,True)
 delta=(scores(b,x)-scores(a,x)).numpy();emp=np.mean([np.maximum(0,(delta[y==k]-delta[y==k,k,None]).max(1)).mean() for k in range(3)])
 assert emp<=c['bound']+1e-9

def test_budget_gain_formula_and_moment_ambiguity():
 for kappa,rho in [(2.,3.),(0.,3.),(2.,0.)]:
  eps=.01;a=2.;L=4.;alpha=2*eps/(kappa+np.sqrt(kappa*kappa+4*rho*eps));alpha=min(alpha,a/L)
  assert kappa*alpha+rho*alpha*alpha<=eps+1e-15
  assert a*alpha-L*alpha*alpha/2>=a*alpha/2-1e-15
 p=np.array([-1.,1.]);q=np.array([0.,0.,-np.sqrt(2),np.sqrt(2)])
 np.testing.assert_allclose([p.mean(),(p*p).mean()],[q.mean(),(q*q).mean()])
 assert np.maximum(p*p-1,0).mean()==0
 np.testing.assert_allclose(np.maximum(q*q-1,0).mean(),.5)

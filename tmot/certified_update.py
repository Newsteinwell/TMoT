"""One-step moment-geometry adaptation; no historical samples enter step()."""
import numpy as np
import torch
from .moments import ClassificationMoments, RegressionMoments
DT=torch.float64

def t(x): return torch.as_tensor(x,dtype=DT)
def macro(y,p): return float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)]))
def cls_stats(memory,classes):
 n=np.array([memory.cells[c].n for c in classes],float)
 mu=np.stack([memory.cells[c].total[:-1]/memory.cells[c].n for c in classes])
 cov=np.stack([memory.cells[c].gram[:-1,:-1]/memory.cells[c].n for c in classes])-mu[:,:,None]*mu[:,None,:]
 return n,mu,(cov+cov.transpose(0,2,1))/2

def head(A,memory,classes,eta=None):
 r=A.shape[0]
 if isinstance(memory,ClassificationMoments):
  n,mu,cov=cls_stats(memory,classes);mu=t(mu);cov=t(cov)
  hmu=mu@A.T;hc=A@cov@A.T;pool=hc.mean(0)
  if eta is None:eta=max(.01*float(torch.trace(pool).detach())/r,1e-8)
  w=t(n/(n+r))[:,None,None]
  regular=w*hc+(1-w)*pool+eta*torch.eye(r,dtype=DT)
  precision=torch.linalg.inv(regular)
  return dict(kind='classification',A=A,means=hmu,precision=precision,classes=classes,eta=eta)
 g,k,_=memory.risk_moments();d=A.shape[1]
 H=torch.zeros((r+1,d+1),dtype=DT);H[:r,:d]=A;H[r,d]=1
 penalty=.1*torch.eye(r+1,dtype=DT);penalty[-1,-1]=0
 weight=torch.linalg.solve(H@t(g)@H.T+penalty,H@t(k))
 return dict(kind='regression',A=A,F=H.T@weight,eta=0.)

def scores(h,x):
 x=t(x)
 if h['kind']=='regression':return torch.cat([x,torch.ones((len(x),1),dtype=DT)],1)@h['F']
 delta=(x@h['A'].T)[:,None,:]-h['means'][None,:,:]
 return -torch.einsum('ncr,crs,ncs->nc',delta,h['precision'],delta)/h['A'].shape[0]

def loss(h,x,y):
 s=scores(h,x)
 if h['kind']=='regression':return (s-t(y)).square().sum(1).mean()
 iy=torch.tensor([h['classes'].index(int(c)) for c in y],dtype=torch.long)
 # Class-balanced current CE, same quadratic logits as inference.
 ce=torch.nn.functional.cross_entropy(s,iy,reduction='none')
 return torch.stack([ce[iy==i].mean() for i in iy.unique()]).mean()

def detach(h):return {k:(v.detach().clone() if isinstance(v,torch.Tensor) else v) for k,v in h.items()}
def regression_risk(h,memory):
 g,k,v=memory.risk_moments();f=h['F'].detach().numpy()
 return float(np.sum(f*(g@f))-2*np.sum(f*k)+v)

def quad(h,V):
 P=h['precision'];W=h['A']@V;r=W.shape[0];p=V.shape[1]
 out=torch.zeros((len(h['classes']),p+1,p+1),dtype=DT)
 out[:,:p,:p]=-(W.T@P@W)/r
 linear=(h['means'][:,None,:]@P@W).squeeze(1)/r
 out[:,:p,p]=linear;out[:,p,:p]=linear
 out[:,p,p]=-torch.einsum('cr,crs,cs->c',h['means'],P,h['means'])/r
 return out

def positive(x):
 e,v=torch.linalg.eigh((x+x.T)/2)
 return (v*e.clamp_min(0))@v.T

def certificate(old,new,memory,spectral=False):
 if isinstance(memory,RegressionMoments):
  diff=regression_risk(new,memory)-regression_risk(old,memory)
  return dict(bound=max(0.,diff),center=max(0.,diff),spectral=None,exact_risk_delta=diff)
 # A QR union can include redundant directions; it contains both row spaces exactly.
 V=torch.linalg.qr(torch.cat([old['A'].T,new['A'].T],1),mode='reduced').Q
 Q=quad(new,V)-quad(old,V);p=V.shape[1];rows=[]
 for c,cell in memory.cells.items():
  i=old['classes'].index(c);D=Q-Q[i];D=torch.cat([D[:i],D[i+1:]],0)
  mu=t(cell.total[:-1]/cell.n)@V
  cov=t(cell.gram[:-1,:-1]/cell.n)-torch.outer(t(cell.total[:-1]/cell.n),t(cell.total[:-1]/cell.n))
  cov=V.T@cov@V;cov=(cov+cov.T)/2;variance=max(0.,float(torch.trace(cov)))
  B=D[:,:p,:p];b=D[:,:p,p]
  values=torch.einsum('p,jpq,q->j',mu,B,mu)+2*(b@mu)+D[:,p,p]
  grad=2*(B@mu+b)
  center=max(0.,float(values.max()))+float(torch.linalg.vector_norm(grad,dim=1).max())*np.sqrt(variance)+float(torch.linalg.matrix_norm(B,ord='fro',dim=(1,2)).max())*variance
  spec=None
  if spectral:
   M=torch.empty((p+1,p+1),dtype=DT);M[:p,:p]=cov+torch.outer(mu,mu);M[:p,p]=mu;M[p,:p]=mu;M[p,p]=1
   e,u=torch.linalg.eigh((M+M.T)/2)
   # PSD roundoff repair only; substantial negatives are errors.
   if float(e.min()) < -1e-9*max(1.,float(e.abs().max())):raise AssertionError('Non-PSD historical moment')
   L=u*e.clamp_min(0).sqrt();S=L.T@D@L
   order=sorted(range(len(S)),key=lambda j:float(torch.linalg.eigvalsh(S[j]).clamp_min(0).sum()),reverse=True)
   H=torch.zeros_like(M)
   for j in order:H=H+positive(S[j]-H)
   # Verify every LMI; add a conservative numerical feasibility correction.
   violation=max(0.,-float(torch.linalg.eigvalsh(H).min()),max(-float(torch.linalg.eigvalsh(H-S[j]).min()) for j in range(len(S))))
   spec=float(torch.trace(H))+(p+1)*(violation+1e-12)
  rows.append(dict(label=int(c),center=center,spectral=spec,bound=min(center,spec) if spec is not None else center))
 return dict(bound=float(np.mean([v['bound'] for v in rows])),center=float(np.mean([v['center'] for v in rows])),spectral=float(np.mean([v['spectral'] for v in rows])) if spectral else None,classes=rows)

def initial(d,r,seed):
 q=np.linalg.qr(np.random.default_rng(seed).normal(size=(d,r)))[0]
 return t(q.T)

def step(A,memory,oldmemory,classes,x,y,mode,budget):
 A=A.detach().clone().requires_grad_(True);ref=head(A,memory,classes);f=loss(ref,x,y);refval=float(f.detach());reference=detach(ref)
 proposals=[]
 if mode=='frozen' or oldmemory is None:return A.detach(),reference,dict(reference_loss=refval,accepted=False,reason='frozen_or_initial',proposals=[]),reference
 f.backward();grad=A.grad;grad=(grad-((grad@A.T+A@grad.T)/2)@A).detach()
 if isinstance(oldmemory,ClassificationMoments):G=oldmemory.risk_moments(classes)[0][:-1,:-1]
 else:G=oldmemory.risk_moments()[0][:-1,:-1]
 reg=max(.01*np.trace(G)/len(G),1e-8)
 V=-torch.linalg.solve(t(G)+reg*torch.eye(len(G),dtype=DT),grad.T).T
 V=(V-((V@A.T+A@V.T)/2)@A).detach()
 norm=float(torch.linalg.vector_norm(V))
 if norm<1e-14:return A.detach(),reference,dict(reference_loss=refval,accepted=False,reason='zero_gradient',proposals=[]),reference
 V=V/norm;a=-float((grad*V).sum());minimum=1e-4*max(1.,refval)
 if a<=0:raise AssertionError('Proposal is not a descent direction')
 chosen=None
 for k in range(12):
  alpha=2.**(-k);Z=A.detach()+alpha*V.detach();U,_,VH=torch.linalg.svd(Z,full_matrices=False);Ac=U@VH
  with torch.no_grad():candidate=head(Ac,memory,classes,reference['eta']);value=float(loss(candidate,x,y))
  cert=certificate(reference,candidate,oldmemory,False)
  gain=refval-value
  if mode=='spectral' and cert['bound']>budget and gain>=max(minimum,1e-4*alpha*a):cert=certificate(reference,candidate,oldmemory,True)
  accept=mode!='curve' and gain>=max(minimum,1e-4*alpha*a) and (mode=='unguarded' or cert['bound']<=budget)
  row=dict(alpha=alpha,gain=gain,current_loss=value,certificate=cert,accepted=accept,head=detach(candidate))
  proposals.append(row)
  if accept:chosen=row;break
 result=dict(reference_loss=refval,minimum_gain=minimum,directional_descent=a,gradient_norm=float(torch.linalg.vector_norm(grad)),accepted=chosen is not None,reason='accepted' if chosen else 'no_effective_feasible_step',proposals=proposals)
 return (chosen['head']['A'],chosen['head'],result,reference) if chosen else (A.detach(),reference,result,reference)

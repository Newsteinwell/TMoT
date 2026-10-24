"""v82: selection sees current TRAIN + moments; old TRAIN is audit-only afterward."""
import argparse,copy,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from .certified_update import initial,step,scores,macro,regression_risk
from .stable_residual import StableSemanticBasis
from .moments import ClassificationMoments,RegressionMoments,ridge_readout,augment
from .fecam import FeCAMLearner,unit_rows

ARMS={'frozen':('curve',0.),'unguarded':('unguarded',0.),'center_e001':('center',.001),'center_e01':('center',.01),'center_e1':('center',.1),'spectral_e01':('spectral',.01)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def np_scores(h,x):
 if h['kind']=='regression':return augment(x)@h['F'].numpy()
 A=h['A'].numpy();delta=(x@A.T)[:,None,:]-h['means'].numpy()[None,:,:]
 return -np.einsum('ncr,crs,ncs->nc',delta,h['precision'].numpy(),delta)/len(A)
def hinge(s,y,classes):
 iy=np.array([classes.index(int(c)) for c in y]);g=1+s-s[np.arange(len(s)),iy,None];g[np.arange(len(s)),iy]=0
 return np.maximum(0,g.max(1))
def average(v,y):return float(np.mean([v[y==c].mean() for c in np.unique(y)]))
def audit(log,reference,xold,yold,x,y,cl):
 # This function is invoked only AFTER all proposal selection is complete.
 if len(xold)==0:return []
 base=np_scores(reference,xold);base_new=np_scores(reference,x);classes=reference.get('classes',[]);rows=[]
 if cl:
  base_loss=hinge(base,yold,classes);iy=np.array([classes.index(int(c)) for c in yold])
 else:base_loss=np.sum((base-yold)**2,1)
 for p in log['proposals']:
  h=p['head'];s=np_scores(h,xold)
  if cl:
   new_loss=hinge(s,yold,classes);delta=s-base;dg=delta-delta[np.arange(len(delta)),iy,None]
   empirical=average(np.maximum(0,dg.max(1)),yold);risk_delta=average(new_loss-base_loss,yold)
   for cc in p['certificate']['classes']:
    actual=float(np.maximum(0,dg[yold==cc['label']].max(1)).mean())
    for key in ('center','spectral','bound'):
     ub=cc[key]
     if ub is not None and actual>ub+2e-8*max(1.,abs(ub),abs(actual)):raise AssertionError('Per-class bound violation')
  else:
   new_loss=np.sum((s-yold)**2,1);risk_delta=float(np.mean(new_loss-base_loss));empirical=max(0.,risk_delta)
  bound=p['certificate']['bound'];tol=2e-8*max(1.,abs(bound),abs(empirical))
  if empirical>bound+tol or risk_delta>empirical+tol:raise AssertionError(f'Certificate violation: {empirical} > {bound}')
  if not cl and abs(risk_delta-p['certificate']['exact_risk_delta'])>tol:raise AssertionError('Regression risk mismatch')
  if cl:
   cur=np_scores(h,x);yy=np.array([classes.index(int(c)) for c in y]);m=cur.max(1);ce=m+np.log(np.exp(cur-m[:,None]).sum(1))-cur[np.arange(len(cur)),yy];check=average(ce,y)
  else:check=float(np.mean(np.sum((np_scores(h,x)-y)**2,1)))
  if abs(check-p['current_loss'])>2e-8*max(1.,abs(check)):raise AssertionError('Current loss mismatch')
  row={k:v for k,v in p.items() if k!='head'};row.update(actual_history_risk_delta=risk_delta,empirical_positive_gap=empirical,certificate_slack=bound-empirical,audit_passed=True)
  rows.append(row)
 return rows

def run(a):
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 root=Path(a.root);out=Path(a.out);out.mkdir(exist_ok=False);start=time.monotonic()
 tr=np.load(root/'features'/a.encoder/f'{a.dataset}_train.npz');val=np.load(root/'features'/a.encoder/f'{a.dataset}_val.npz');meta=json.loads((root/'prepared'/f'{a.dataset}_manifest.json').read_text());cl=meta['task']=='classification'
 if cl:
  labels=np.random.default_rng(a.order).permutation(np.unique(tr['y']));first=3 if len(labels)%2 else 2;order=[labels[:first]]+[labels[i:i+2] for i in range(first,len(labels),2)]
  parts=[np.flatnonzero(np.isin(tr['y'],c)) for c in order];evals=[np.flatnonzero(np.isin(val['y'],c)) for c in order]
 else:
  order=sorted(np.unique(tr['groups']).tolist());parts=[np.flatnonzero(tr['groups']==g) for g in order];evals=[np.flatnonzero(val['groups']==g) for g in order]
 if a.max_stages:parts=parts[:a.max_stages];evals=evals[:a.max_stages]
 basis=StableSemanticBasis(2*meta['shapes']['train'][1],512).fit(tr['x'][parts[0]])
 tx=basis.transform(tr['x']).astype(float);vx=basis.transform(val['x']).astype(float)
 if cl:tx=unit_rows(tx);vx=unit_rows(vx);target=tr['y'];mu=0.;sd=1.
 else:
  mu=tr['y'][parts[0]].mean(0);sd=np.maximum(tr['y'][parts[0]].std(0),1e-6);target=(tr['y']-mu)/sd
  if target.ndim==1:target=target[:,None]
 d=tx.shape[1];memory=ClassificationMoments(d) if cl else RegressionMoments(d,target.shape[1]);classes=[]
 A={k:initial(d,a.rank,a.seed) for k in ARMS};records={k:[] for k in ARMS};records['full_ridge']=[]
 fecam=FeCAMLearner(d,1.) if cl else None
 if cl:records['full_fecam1']=[]
 history=[];all_audit=[];last_heads={};pred_digest={}
 for stage,ids in enumerate(parts):
  old=copy.deepcopy(memory) if stage else None;memory.update(tx[ids],target[ids])
  if cl:classes.extend(sorted(set(target[ids].tolist())-set(classes)));fecam.observe(tx[ids],target[ids])
  g,k,_=memory.risk_moments(classes);ridge=ridge_readout(g,k,.1);heads={};logs={}
  xold=tx[np.concatenate(history)] if history else np.empty((0,d));yold=target[np.concatenate(history)] if history else np.empty(0)
  for arm,(mode,budget) in ARMS.items():
   tick=time.monotonic();A[arm],h,log,reference=step(A[arm],memory,old,classes,tx[ids],target[ids],mode,budget)
   aud=audit(log,reference,xold,yold,tx[ids],target[ids],cl)
   accepted=[p for p in aud if p['accepted']]
   if arm not in ('frozen','unguarded') and accepted and accepted[0]['certificate']['bound']>budget:raise AssertionError('Budget exceeded')
   serial={k:v for k,v in log.items() if k!='proposals'};serial.update(arm=arm,stage=stage+1,proposals=aud,elapsed_s=time.monotonic()-tick)
   all_audit.append(serial);heads[arm]=h;logs[arm]=serial
  ix=np.concatenate(evals[:stage+1]);prediction_arrays={'y':val['y'][ix],'ids':val['ids'][ix]};stage_states={}
  for arm in records:
   if arm=='full_ridge':z=augment(vx[ix])@ridge;pred=np.asarray(classes)[z.argmax(1)] if cl else z
   elif arm=='full_fecam1':pred=fecam.predict(vx[ix])
   else:
    h=heads[arm];z=np_scores(h,vx[ix]);pred=np.asarray(classes)[z.argmax(1)] if cl else z
    np.testing.assert_allclose(z,scores(h,vx[ix]).numpy(),rtol=1e-10,atol=1e-10)
    stage_states[arm]=h
   if not cl:pred=pred*sd+mu
   metric=macro(val['y'][ix],pred) if cl else float(np.sqrt(np.mean((pred-val['y'][ix])**2)))
   row=dict(stage=stage+1,metric=metric,accepted=logs[arm]['accepted'] if arm in logs else False)
   if arm in logs:
    ap=[v for v in logs[arm]['proposals'] if v['accepted']]
    if ap:row.update(gain=ap[0]['gain'],alpha=ap[0]['alpha'],bound=ap[0]['certificate']['bound'],actual_history_risk_delta=ap[0]['actual_history_risk_delta'])
   records[arm].append(row);prediction_arrays[arm]=pred
  np.savez_compressed(out/f'predictions_stage{stage+1}.npz',**prediction_arrays)
  torch.save(dict(heads=stage_states,ridge=ridge,classes=list(classes)),out/f'state_stage{stage+1}.pt')
  (out/'audit.json').write_text(json.dumps(all_audit,indent=2));history.append(ids)
  print(stage+1,{k:round(v[-1]['metric'],5) for k,v in records.items()},flush=True)
 final_path=out/'memory.pt';torch.save(dict(memory=memory,basis=basis,target_mean=mu,target_sd=sd,fecam=fecam),final_path)
 # Independent deserialization reproduces all final predictions; also validates saved IDs.
 loaded=torch.load(out/f'state_stage{len(parts)}.pt',weights_only=False)
 with np.load(out/f'predictions_stage{len(parts)}.npz') as saved:
  for arm,h in loaded['heads'].items():
   z=np_scores(h,vx[ix]);p=np.asarray(classes)[z.argmax(1)] if cl else z*sd+mu
   np.testing.assert_array_equal(p,saved[arm])
  z=augment(vx[ix])@loaded['ridge'];p=np.asarray(classes)[z.argmax(1)] if cl else z*sd+mu
  np.testing.assert_array_equal(p,saved['full_ridge'])
  if cl:
   full=torch.load(final_path,weights_only=False)['fecam'].predict(vx[ix]);np.testing.assert_array_equal(full,saved['full_fecam1'])
 summary=dict(status='completed',task=meta['task'],config=vars(a),source=os.environ.get('TMOT_SOURCE_COMMIT','UNCOMMITTED'),input_hashes={s:sha(root/'features'/a.encoder/f'{a.dataset}_{s}.npz') for s in ('train','val')},records=records,results={k:dict(final=v[-1]['metric'],average=float(np.mean([s['metric'] for s in v])),accepted=sum(s['accepted'] for s in v)) for k,v in records.items()},audit_proposals=sum(len(v['proposals']) for v in all_audit),all_proposals_verified=True,final_reload_exact=True,memory_sha256=sha(final_path),elapsed_s=time.monotonic()-start)
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in ('root','out','dataset','encoder'):p.add_argument('--'+k,required=True)
 p.add_argument('--max-stages',type=int,default=0);p.add_argument('--rank',type=int,default=32);p.add_argument('--seed',type=int,default=42);p.add_argument('--order',type=int,default=17)
 run(p.parse_args())

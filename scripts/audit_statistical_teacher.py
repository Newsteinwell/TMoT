"""Independent class covariance / regression normal equation and student audit."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def metric(y,p,cl):return float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)])) if cl else float(np.sqrt(np.mean((y-p)**2)))
def aug(x):return np.column_stack([np.asarray(x,dtype=np.float64),np.ones(len(x))])
def unit(x):
 x=np.asarray(x,dtype=np.float64);return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
root=Path(sys.argv[1]);q=root/'runs/statistical_teacher_v74';torch.set_num_threads(1);reports=[]
for basepath in sorted(q.glob('*_kd0/summary.json')):
 s=json.loads(basepath.read_text());a=s['config'];d,e=a['dataset'],a['encoder'];cl=s['task']=='classification';tr=np.load(root/'features'/e/f'{d}_train.npz');va=np.load(root/'features'/e/f'{d}_val.npz');lookup={str(v):i for i,v in enumerate(va['ids'])};checkpoint=torch.load(basepath.parent/'final.pt',map_location='cpu',weights_only=False);basis=checkpoint['basis'];mu=checkpoint['target_mean'];sd=checkpoint['target_sd'];classes=[];means=[];precisions=[];gram=np.zeros((513,513));cross=None;n=0;teacher_stages=[]
 def distances(x):
  z=unit(x);return np.column_stack([np.einsum('ni,ni->n',(z-m)@p,z-m) for m,p in zip(means,precisions)])
 for t,part in enumerate(s['order']):
  ids=np.flatnonzero(np.isin(tr['y'],part)) if cl else np.flatnonzero(tr['groups']==part)
  z=basis.transform(tr['x'][ids]);targets=tr['y'][ids] if cl else (tr['y'][ids]-mu)/sd
  if cl:
   for c in sorted(part):
    v=unit(z[targets==c]);m=v.mean(0);delta=v-m;cov=delta.T@delta/max(len(v)-1,1);cov+=10*max(np.trace(cov)/512,1e-12)*np.eye(512);scale=np.sqrt(np.diag(cov));cor=cov/np.outer(scale,scale);means.append(m);precisions.append(np.linalg.solve(cor,np.eye(512)));classes.append(c)
   dist=distances(z);logits=-dist/np.maximum(dist.mean(1,keepdims=True),1e-12);logits-=logits.max(1,keepdims=True);soft=np.exp(logits);soft/=soft.sum(1,keepdims=True)
  else:
   zz=aug(z);yy=np.asarray(targets,dtype=np.float64).reshape(len(z),-1);gram+=zz.T@zz;cross=zz.T@yy if cross is None else cross+zz.T@yy;n+=len(z);g=gram/n;penalty=np.eye(513)*.1;penalty[-1,-1]=0;readout=np.linalg.solve((g+g.T)/2+penalty,cross/n);soft=zz@readout
  saved=np.load(basepath.parent/f'teacher_train_stage{t+1}.npz');np.testing.assert_array_equal(saved['ids'],tr['ids'][ids]);np.testing.assert_allclose(saved['targets'],soft,rtol=1e-8,atol=1e-9)
  if cl:np.testing.assert_array_equal(saved['classes'],classes)
  vp=np.load(basepath.parent/f'predictions_stage{t+1}.npz');ys=[];ps=[]
  for k in range(t+1):
   inds=np.array([lookup[str(i)] for i in vp[f'ids_{k}']]);vz=basis.transform(va['x'][inds]);actual=np.asarray(classes)[distances(vz).argmin(1)] if cl else (aug(vz)@readout)*sd+mu
   np.testing.assert_allclose(vp[f'teacher_{k}'],actual,rtol=1e-8,atol=1e-9);ys.append(va['y'][inds]);ps.append(actual)
   if cl:
    old=root/f'runs/fecam_harder_v59/{d}_{e}_fecam_a10_o17_s42/predictions_stage{t+1}.npz';oldp=np.load(old);np.testing.assert_array_equal(vp[f'teacher_{k}'],oldp[f'pred_{k}']);np.testing.assert_array_equal(vp[f'ids_{k}'],oldp[f'ids_{k}'])
  teacher_stages.append(metric(np.concatenate(ys),np.concatenate(ps),cl))
 for strength in (0.,.1,1.,10.):
  folder=q/f'{d}_{e}_kd{strength:g}';ss=json.loads((folder/'summary.json').read_text());model=torch.load(folder/'final.pt',map_location='cpu',weights_only=False);values=[];lastys=[];lastps=[]
  for split in ('train','val'):assert sha(root/'features'/e/f'{d}_{split}.npz')==ss['input_hashes'][split]
  for row in ss['stages']:
   t=row['stage'];p=np.load(folder/f'predictions_stage{t}.npz');reference=np.load(basepath.parent/f'predictions_stage{t}.npz');soft=np.load(folder/f'teacher_train_stage{t}.npz');refsoft=np.load(basepath.parent/f'teacher_train_stage{t}.npz')
   for key in soft.files:np.testing.assert_array_equal(soft[key],refsoft[key])
   ys=[];ps=[]
   for k in range(t):
    inds=np.array([lookup[str(i)] for i in p[f'ids_{k}']]);y=va['y'][inds];pred=p[f'pred_{k}'];np.testing.assert_array_equal(y,p[f'y_{k}']);np.testing.assert_array_equal(p[f'teacher_{k}'],reference[f'teacher_{k}']);ys.append(y);ps.append(pred)
    np.testing.assert_allclose(metric(y,pred,cl),row['scores'][k],rtol=0,atol=1e-12)
    if strength==0:
     old=np.load(root/f'runs/sequential_head_v73/{d}_{e}_kd0/predictions_stage{t}.npz');np.testing.assert_array_equal(old[f'pred_{k}'],pred)
    if t==len(ss['stages']):
     actual=model['model'].predict_head(model['basis'].transform(va['x'][inds]));actual=actual if cl else actual*sd+mu;np.testing.assert_array_equal(actual,pred)
   lastys=np.concatenate(ys);lastps=np.concatenate(ps);value=metric(lastys,lastps,cl);name='macro_accuracy' if cl else 'rmse';np.testing.assert_allclose(value,row[name],rtol=0,atol=1e-12);values.append(value)
  np.testing.assert_allclose([values[-1],np.mean(values)],[ss['final_'+name],ss['average_'+name]],rtol=0,atol=1e-12)
  r=dict(dataset=d,encoder=e,task=ss['task'],strength=strength,final=values[-1],average=float(np.mean(values)),teacher_final=teacher_stages[-1],teacher_average=float(np.mean(teacher_stages)),source=ss['git_commit'],teacher_bytes=ss['stages'][-1]['teacher_persistent_bytes'])
  if cl:
   new=np.asarray(ss['order'][-1]);old=~np.isin(lastys,new);r.update(old_macro=metric(lastys[old],lastps[old],True),old_predicted_as_new=float(np.mean(np.isin(lastps[old],new))))
  reports.append(r)
 print(d,e,'audited',flush=True)
assert len(reports)==48
(q/'audit.json').write_text(json.dumps(dict(status='passed',runs=reports,checks=['FT trajectories exact v73','classification teacher exact v59 alpha10','independent teacher targets and predictions','all metrics and sample IDs','final student checkpoint exact']),indent=2));print('48 trajectories passed')

"""Verify support regeneration, quotas, fixed-teacher controls and predictions."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from tmot.distillation_support import DistillationSupport

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def metric(y,p,cl):return float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)])) if cl else float(np.sqrt(np.mean((y-p)**2)))
root=Path(sys.argv[1]);q=root/'runs/target_confidence_v77';torch.set_num_threads(1);reports=[]
for f in sorted(q.glob('*/summary.json')):
 s=json.loads(f.read_text());a=s['config'];d,e,mode=a['dataset'],a['encoder'],a['support'];cl=s['task']=='classification';name='macro_accuracy' if cl else 'rmse';tr=np.load(root/'features'/e/f'{d}_train.npz');va=np.load(root/'features'/e/f'{d}_val.npz');lookup={str(i):k for k,i in enumerate(va['ids'])};state=torch.load(f.parent/'final.pt',map_location='cpu',weights_only=False);basis=state['basis'];model=state['model'];mu=state['target_mean'];sd=state['target_sd'];pool=DistillationSupport(512,s['task'],mode);values=[];teachers=[];quota=0
 for split in ('train','val'):assert s['input_hashes'][split]==sha(root/'features'/e/f'{d}_{split}.npz')
 for t,(part,row) in enumerate(zip(s['order'],s['stages'])):
  ids=np.flatnonzero(np.isin(tr['y'],part)) if cl else np.flatnonzero(tr['groups']==part);x=basis.transform(tr['x'][ids]);y=tr['y'][ids] if cl else (tr['y'][ids]-mu)/sd
  hx,info=pool.sample(x,y);assert info==row['support'];reference=json.loads((root/f'runs/support_teacher_v75/{d}_{e}_local/summary.json').read_text());assert info==reference['stages'][t]['support'];assert row['optimizer_steps']==30*int(np.ceil((len(ids)+4096)/256));assert row['warm_init_error']==0
  if mode=='er10' and info['historical_support']:
   # Exact row-membership in the pre-arrival buffer; current rows not yet stored.
   buffered={v.tobytes() for v in pool.real.buffer[0]};assert all(v.tobytes() in buffered for v in hx)
  pool.update(x,y);assert pool.nbytes==row['support_memory_bytes'];assert pool.audit==row['buffer_audit']
  if mode=='er10':
   audit=row['buffer_audit'];quota+=len(ids)//10;chosen=audit['selected_indices'];assert audit['after']==quota==row['stored_examples'];assert len(chosen)==len(set(chosen))==len(ids)//10;assert all(0<=i<len(ids) for i in chosen)
  else:assert row['stored_examples']==0
  p=np.load(f.parent/f'predictions_stage{t+1}.npz');teacher=np.load(root/f'runs/statistical_teacher_v74/{d}_{e}_kd0/predictions_stage{t+1}.npz');ys=[];ps=[];ts=[]
  control=np.load(root/f'runs/support_teacher_v75/{d}_{e}_local/predictions_stage{t+1}.npz') if a['target_mode']=='soft' else None
  for k in range(t+1):
   vi=np.array([lookup[str(i)] for i in p[f'ids_{k}']]);truth=va['y'][vi];pred=p[f'pred_{k}'];np.testing.assert_array_equal(truth,p[f'y_{k}']);np.testing.assert_array_equal(p[f'ids_{k}'],teacher[f'ids_{k}']);np.testing.assert_array_equal(p[f'teacher_{k}'],teacher[f'teacher_{k}']);ys.append(truth);ps.append(pred);ts.append(p[f'teacher_{k}'])
   np.testing.assert_allclose(metric(truth,pred,cl),row['scores'][k],rtol=0,atol=1e-12)
   if control is not None:np.testing.assert_array_equal(pred,control[f'pred_{k}'])
   if t==len(s['stages'])-1:
    actual=model.predict_head(basis.transform(va['x'][vi]));actual=actual if cl else actual*sd+mu;np.testing.assert_array_equal(actual,pred)
  yy=np.concatenate(ys);pp=np.concatenate(ps);value=metric(yy,pp,cl);np.testing.assert_allclose(value,row[name],rtol=0,atol=1e-12);values.append(value);teachers.append(metric(yy,np.concatenate(ts),cl))
 np.testing.assert_allclose([values[-1],np.mean(values)],[s['final_'+name],s['average_'+name]],rtol=0,atol=1e-12)
 result=dict(target_mode=a['target_mode'],dataset=d,encoder=e,task=s['task'],support=mode,final=values[-1],average=float(np.mean(values)),teacher_final=teachers[-1],teacher_average=float(np.mean(teachers)),support_bytes=pool.nbytes,stored_examples=s['stages'][-1]['stored_examples'],source=s['git_commit'])
 if cl:
  new=np.asarray(s['order'][-1]);old=~np.isin(yy,new);result.update(old_macro=metric(yy[old],pp[old],True),old_predicted_as_new=float(np.mean(np.isin(pp[old],new))))
 reports.append(result);print(d,e,mode,'audited',flush=True)
assert len(reports)==24
(q/'audit.json').write_text(json.dumps(dict(status='passed',runs=reports),indent=2));print('24 trajectories passed')

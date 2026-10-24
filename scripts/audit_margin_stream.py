import hashlib,json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);folder=root/'runs/margin_stream_v67';terminal=json.loads((folder/'complete.json').read_text());assert len(terminal)==36 and all(r['status']=='completed' for r in terminal)
rows=[]
for path in sorted(folder.glob('*/summary.json')):
 s=json.loads(path.read_text());c=s['config'];group=f"{c['dataset']}_{c['encoder']}_o{c['order']}_s{c['seed']}";mode=c['mode'];shared=Path(s['shared_state'])
 with shared.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==s['shared_state_sha256']
 assert s['final_training_state_exact']
 old=root/'runs/bounded_robustness_v64'/f"{group}_{'bounded' if mode=='selector' else mode}";oldsummary=json.loads((old/'summary.json').read_text())
 assert s['frozen']==oldsummary['frozen'] and s['order']==oldsummary['order'] and s['input_hashes']==oldsummary['input_hashes']
 assert c['parameter']==(None if mode=='selector' else s['frozen'][mode])
 for split,h in s['input_hashes'].items():
  with (root/'features'/c['encoder']/f"{c['dataset']}_{split}.npz").open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
 for stage in s['stages']:
  t=stage['stage']
  with np.load(path.parent/f'predictions_stage{t}.npz') as data,np.load(old/f'predictions_stage{t}.npz') as control:
   for j in range(t):
    for key in ('y','ids'):np.testing.assert_array_equal(data[f'{key}_{j}'],control[f'{key}_{j}'])
    if mode!='selector':np.testing.assert_array_equal(data[f'pred_{j}'],control[f'pred_{j}'])
   y=np.concatenate([data[f'y_{j}'] for j in range(t)]);pred=np.concatenate([data[f'pred_{j}'] for j in range(t)])
   score=np.mean([np.mean(pred[y==label]==label) for label in np.unique(y)]);np.testing.assert_allclose(score,stage['macro_accuracy'],atol=1e-12)
   for j in range(t):
    yy=data[f'y_{j}'];pp=data[f'pred_{j}'];np.testing.assert_allclose(np.mean([np.mean(pp[yy==label]==label) for label in np.unique(yy)]),stage['scores'][j],atol=1e-12)
   if mode=='selector':
    margins=[]
    for key in ('reference_distances','candidate_distances'):
     d=np.sort(data[key],axis=1);den=d[:,0]+d[:,1];margins.append(np.divide(d[:,1]-d[:,0],den,out=np.zeros(len(d)),where=den>0))
    choose=margins[1]>margins[0];np.testing.assert_array_equal(choose,data['choose_candidate'])
    rp=data['classes'][data['reference_distances'].argmin(1)];cp=data['classes'][data['candidate_distances'].argmin(1)]
    np.testing.assert_array_equal(cp,np.concatenate([control[f'pred_{j}'] for j in range(t)]))
    with np.load(root/'runs/bounded_robustness_v64'/f'{group}_fecam'/f'predictions_stage{t}.npz') as fecam:np.testing.assert_array_equal(rp,np.concatenate([fecam[f'pred_{j}'] for j in range(t)]))
    np.testing.assert_array_equal(pred,np.where(choose,cp,rp))
    if t==len(s['order']):
     with np.load(root/'runs/margin_selector_v66'/f'{group}.npz') as saved:
      np.testing.assert_array_equal(pred,saved['prediction'])
      for key in ('reference_distances','candidate_distances','choose_candidate'):np.testing.assert_array_equal(data[key],saved[key])
 np.testing.assert_allclose(s['final_macro_accuracy'],s['stages'][-1]['macro_accuracy']);np.testing.assert_allclose(s['average_macro_accuracy'],np.mean([r['macro_accuracy'] for r in s['stages']]))
 rows.append(s)
assert len(rows)==216
(root/'margin_stream_v67_runs.json').write_text(json.dumps(rows,indent=2))
pairs=[];summary=[]
for r in rows:
 c=r['config']
 if c['mode']!='selector':continue
 for ref in ('fecam','ridge','original','bounded'):
  peer=next(p for p in rows if all(p['config'][k]==c[k] for k in ('dataset','encoder','seed','order')) and p['config']['mode']==ref)
  pairs.append(dict(dataset=c['dataset'],encoder=c['encoder'],seed=c['seed'],order=c['order'],reference=ref,development_control=(c['seed'],c['order'])==(42,17),final_gain_pp=100*(r['final_macro_accuracy']-peer['final_macro_accuracy']),average_gain_pp=100*(r['average_macro_accuracy']-peer['average_macro_accuracy'])))
for key in sorted({(p['dataset'],p['encoder'],p['reference']) for p in pairs}):
 selected=[p for p in pairs if (p['dataset'],p['encoder'],p['reference'])==key and not p['development_control']];final=[p['final_gain_pp'] for p in selected];avg=[p['average_gain_pp'] for p in selected]
 item=dict(zip(('dataset','encoder','reference'),key));item.update(n=len(selected),mean_final_gain_pp=float(np.mean(final)),min_final_gain_pp=min(final),max_final_gain_pp=max(final),mean_average_gain_pp=float(np.mean(avg)),min_average_gain_pp=min(avg),max_average_gain_pp=max(avg),final_wins=sum(v>1e-10 for v in final),average_wins=sum(v>1e-10 for v in avg));summary.append(item)
 if key[2]=='fecam':print(json.dumps(item),flush=True)
(root/'margin_stream_v67_comparison.json').write_text(json.dumps(dict(complete=True,trajectories=216,pairs=pairs,summary=summary,checks=['exact v64 controls every stage','independent margin selection every stage','sample identities and metrics','exact v66 final scores and predictions','frozen parameters and source hashes']),indent=2))

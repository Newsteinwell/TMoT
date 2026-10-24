"""Read-only error decomposition, never a deployable label-based selector."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def decomposition(y,reference,candidate):
 y=np.asarray(y);reference=np.asarray(reference);candidate=np.asarray(candidate)
 assert y.shape==reference.shape==candidate.shape and len(y)>0
 labels=np.unique(y);weights=np.array([1/(len(labels)*np.sum(y==v)) for v in y])
 good_ref=reference==y;good_cand=candidate==y
 out={name:float(weights[mask].sum()) for name,mask in dict(both_correct=good_ref&good_cand,rescue=(~good_ref)&good_cand,harm=good_ref&(~good_cand),both_wrong=(~good_ref)&(~good_cand),disagree=reference!=candidate).items()}
 out.update(reference_accuracy=out['both_correct']+out['harm'],candidate_accuracy=out['both_correct']+out['rescue'],label_oracle_selector_accuracy=1-out['both_wrong'])
 assert abs(out['both_correct']+out['rescue']+out['harm']+out['both_wrong']-1)<1e-12
 return out

def run(root):
 source=root/'bounded_robustness_v64_runs.json';runs=json.loads(source.read_text());folder=root/'runs/bounded_robustness_v64';rows=[]
 for s in runs:
  c=s['config']
  if c['mode'] not in ('bounded','original'):continue
  prefix=f"{c['dataset']}_{c['encoder']}_o{c['order']}_s{c['seed']}"
  for reference in ('fecam','ridge'):
   peer=next(p for p in runs if all(p['config'][k]==c[k] for k in ('dataset','encoder','seed','order')) and p['config']['mode']==reference)
   stages=[]
   for stage in range(1,len(s['stages'])+1):
    with np.load(folder/(prefix+'_'+c['mode'])/f'predictions_stage{stage}.npz') as pred,np.load(folder/(prefix+'_'+reference)/f'predictions_stage{stage}.npz') as ref:
     for k in range(stage):
      for key in ('y','ids'):np.testing.assert_array_equal(pred[f'{key}_{k}'],ref[f'{key}_{k}'])
     y=np.concatenate([pred[f'y_{k}'] for k in range(stage)]);pp=np.concatenate([pred[f'pred_{k}'] for k in range(stage)]);rr=np.concatenate([ref[f'pred_{k}'] for k in range(stage)])
     row=decomposition(y,rr,pp);row['stage']=stage
     np.testing.assert_allclose(row['candidate_accuracy'],s['stages'][stage-1]['macro_accuracy'],atol=1e-12)
     np.testing.assert_allclose(row['reference_accuracy'],peer['stages'][stage-1]['macro_accuracy'],atol=1e-12)
     stages.append(row)
   rows.append(dict(dataset=c['dataset'],encoder=c['encoder'],seed=c['seed'],order=c['order'],candidate=c['mode'],reference=reference,development_control=(c['seed'],c['order'])==(42,17),stages=stages))
 summary=[]
 for key in sorted({tuple(r[k] for k in ('dataset','encoder','candidate','reference')) for r in rows}):
  selected=[r for r in rows if tuple(r[k] for k in ('dataset','encoder','candidate','reference'))==key and not r['development_control']]
  item=dict(zip(('dataset','encoder','candidate','reference'),key));item['n']=len(selected)
  for metric in ('rescue','harm','both_wrong','disagree','reference_accuracy','candidate_accuracy','label_oracle_selector_accuracy'):
   item['final_'+metric]=float(np.mean([r['stages'][-1][metric] for r in selected]))
   item['average_'+metric]=float(np.mean([np.mean([s[metric] for s in r['stages']]) for r in selected]))
  summary.append(item)
 assert len(rows)==144 and len(summary)==24
 result=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),rows=rows,summary=summary,note='Read-only development VAL error decomposition. Label oracle is a diagnostic ceiling only for selecting between these two hard predictions; not an implementable method or a ceiling for general fusion. No TEST, no fitting, no checkpoint mutation.')
 (root/'readout_complementarity_v65.json').write_text(json.dumps(result,indent=2))
 for item in summary:
  if item['candidate']=='bounded' and item['reference']=='fecam':print(json.dumps(item),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);run(p.parse_args().root)

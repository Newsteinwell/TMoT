"""Fixed five-projection voting/averaging diagnostic using saved VAL predictions."""
import argparse,json,collections
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
folder=a.root/'runs/projection_seed_v23';groups=collections.defaultdict(list)
for path in folder.glob('*/summary.json'):
    s=json.loads(path.read_text());groups[s['config']['dataset'],s['config']['dimension']].append((path.parent,s))
assert len(groups)==14 and all(len(v)==5 for v in groups.values())
rows=[]
for (dataset,dimension),runs in sorted(groups.items()):
    runs.sort(key=lambda x:x[1]['config']['basis_seed']);ref=runs[0][1];classification=ref['task']=='classification';records=[]
    assert [s['config']['basis_seed'] for _,s in runs]==list(range(1701,1706))
    assert all(s['order']==ref['order'] and s['input_hashes']==ref['input_hashes'] for _,s in runs)
    for stage in ref['stages']:
        name=f"predictions_stage{stage['stage']}.npz";preds=[];truth=None
        for run,s in runs:
            with np.load(run/name) as d:
                keys=sorted(int(k.split('_')[1]) for k in d.files if k.startswith('y_'))
                y=np.concatenate([d[f'y_{k}'] for k in keys]);ids=np.concatenate([d[f'ids_{k}'] for k in keys])
                if truth is None:truth=y;reference_ids=ids
                else:np.testing.assert_array_equal(truth,y);np.testing.assert_array_equal(reference_ids,ids)
                preds.append(np.concatenate([d[f'pred_{k}'] for k in keys]))
        predictions=np.stack(preds)
        if classification:
            labels=np.sort(np.concatenate(ref['order'][:stage['stage']]));counts=np.stack([(predictions==label).sum(0) for label in labels]);pred=labels[counts.argmax(0)]
            ties=int(np.sum((counts==counts.max(0)).sum(0)>1))
            value=float(np.mean([np.mean(pred[truth==c]==c) for c in labels]))
        else:
            pred=predictions.mean(0);ties=0;value=float(np.sqrt(np.mean((pred-truth)**2)))
        records.append(dict(stage=stage['stage'],value=value,vote_ties=ties,n=len(truth),persistent_bytes=sum(s['stages'][stage['stage']-1]['persistent_bytes'] for _,s in runs)))
    rows.append(dict(dataset=dataset,dimension=dimension,metric='macro_accuracy' if classification else 'rmse',final=records[-1]['value'],average=float(np.mean([r['value'] for r in records])),stages=records))
a.out.write_text(json.dumps(dict(scope='VAL only, fixed five-member ensemble; smallest seen class label breaks ties; no weight selection; summed learner states exclude shared encoder and basis',rows=rows),indent=2))
for r in rows:print(r['dataset'],r['dimension'],r['final'],r['average'])

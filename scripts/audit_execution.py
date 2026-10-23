"""Read-only consistency audit of produced files; never changes model selection."""
import argparse,importlib.metadata,json,platform
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
report={'environment':{'python':platform.python_version()},'datasets':{},'features':{},'queues':{}}
for pkg in ('torch','numpy','scipy','mantis-tsfm','huggingface-hub','pytest'):
    report['environment'][pkg]=importlib.metadata.version(pkg)
for f in sorted((a.root/'prepared').glob('*_manifest.json')):
    report['datasets'][f.stem]=json.loads(f.read_text())
for encoder in ('mantis','summary'):
    files=list((a.root/'features'/encoder).glob('*.json'))
    report['features'][encoder]=[json.loads(f.read_text()) for f in sorted(files)]
for queue in sorted((a.root/'runs').glob('pilot*')):
    if not(queue/'queue_manifest.json').exists():continue
    manifest=json.loads((queue/'queue_manifest.json').read_text());expected=len(manifest['jobs'])
    checked=[]
    for f in sorted(queue.glob('*/summary.json')):
        s=json.loads(f.read_text());assert (f.parent/'final.pt').exists()
        for stage in s['stages']:
            d=np.load(f.parent/f"predictions_stage{stage['stage']}.npz")
            ys=np.concatenate([d[k] for k in sorted(d.files) if k.startswith('y_')])
            ps=np.concatenate([d[k] for k in sorted(d.files) if k.startswith('pred_')])
            if s['task']=='classification':
                metric=np.mean([np.mean(ps[ys==c]==c) for c in np.unique(ys)])
                np.testing.assert_allclose(metric,stage['macro_accuracy'],atol=1e-12)
            else:np.testing.assert_allclose(np.sqrt(np.mean((ys-ps)**2)),stage['rmse'],rtol=1e-6)
        checked.append(f.parent.name)
    statuses=[json.loads(l) for l in (queue/'queue_status.jsonl').read_text().splitlines()] if (queue/'queue_status.jsonl').exists() else []
    report['queues'][queue.name]={'expected':expected,'completed_metric_checked':len(checked),'checked':checked,
                                'failed':[r for r in statuses if r['status'] not in ('completed','already_completed')]}
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2))
print(json.dumps({k:{i:v for i,v in x.items() if i!='checked'} for k,x in report['queues'].items()},indent=2))

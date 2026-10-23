"""Report each fixed-loss paired warm/cold comparison without selecting winners."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('out',type=Path);a=p.parse_args();rows=[]
for path in sorted(a.runs.glob('*_warm0_*/summary.json')):
    cold=json.loads(path.read_text());warm=json.loads((a.runs/path.parent.name.replace('_warm0_','_warm1_')/'summary.json').read_text());c=cold['config']
    cl=cold['task']=='classification';metric='final_macro_accuracy' if cl else 'final_rmse';average='average_macro_accuracy' if cl else 'average_rmse'
    rows.append(dict(dataset=c['dataset'],encoder=c['encoder'],method=c['method'],components=c.get('local_components'),loss=c['residual_loss'],metric=metric,cold=cold[metric],warm=warm[metric],cold_average=cold[average],warm_average=warm[average],improvement=(warm[metric]-cold[metric])*100 if cl else (cold[metric]-warm[metric])/cold[metric]*100))
assert len(rows)==36
a.out.write_text(json.dumps(dict(scope='paired single-seed/order VAL diagnostics, no TEST',rows=rows),indent=2))
for r in rows:print(r['dataset'],r['encoder'],r['method'],r['loss'],round(r['cold'],5),round(r['warm'],5),round(r['improvement'],3))

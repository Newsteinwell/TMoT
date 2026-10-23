"""Report all fixed-loss merge pairs without selecting dataset-wise winners."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('out',type=Path);a=p.parse_args();rows=[]
for path in sorted(a.runs.glob('*_min1_*/summary.json')):
    high=json.loads(path.read_text());low=json.loads((a.runs/path.parent.name.replace('_min1_','_min4_')/'summary.json').read_text());c=high['config']
    cl=high['task']=='classification';metric='final_macro_accuracy' if cl else 'final_rmse';avg='average_macro_accuracy' if cl else 'average_rmse'
    rows.append(dict(dataset=c['dataset'],encoder=c['encoder'],loss=c['residual_loss'],warm=c['warm_start'],metric=metric,min1=high[metric],min4=low[metric],average1=high[avg],average4=low[avg],bytes1=high['stages'][-1]['persistent_bytes'],bytes4=low['stages'][-1]['persistent_bytes']))
assert len(rows)==36
a.out.write_text(json.dumps(dict(scope='Single-seed/order VAL paired screen; no TEST',rows=rows),indent=2))
for r in rows:print(r['dataset'],r['encoder'],r['loss'],r['warm'],round(r['min1'],5),round(r['min4'],5))

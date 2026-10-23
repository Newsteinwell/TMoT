"""Retain every group; deterministic VAL selection breaks ties with smaller k."""
import argparse,json
from pathlib import Path
from collections import defaultdict
p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('out',type=Path);a=p.parse_args();groups=defaultdict(list)
for path in a.runs.glob('*/summary.json'):
    s=json.loads(path.read_text());c=s['config'];groups[c['dataset'],c['encoder'],c['method']].append(s)
assert len(groups)==42 and all(len(v)==5 for v in groups.values())
rows=[]
for key,runs in sorted(groups.items()):
    cl=runs[0]['task']=='classification';metric='final_macro_accuracy' if cl else 'final_rmse';average='average_macro_accuracy' if cl else 'average_rmse'
    best=min(runs,key=lambda s:((-1 if cl else 1)*s[metric],s['config']['neighbors']))
    rows.append(dict(dataset=key[0],encoder=key[1],method=key[2],metric=metric,final=best[metric],average=best[average],k=best['config']['neighbors'],buffer_bytes=best['stages'][-1]['persistent_bytes']))
a.out.write_text(json.dumps(rows,indent=2))

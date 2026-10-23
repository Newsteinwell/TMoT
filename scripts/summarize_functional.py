import argparse,json
from pathlib import Path
from collections import defaultdict
p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('out',type=Path);a=p.parse_args();groups=defaultdict(list)
for path in a.runs.glob('*/summary.json'):
    s=json.loads(path.read_text());c=s['config'];groups[c['dataset'],c['encoder'],c['function_mode']].append(s)
assert len(groups)==18 and all(len(v)==6 for v in groups.values())
rows=[]
for key,runs in sorted(groups.items()):
    cl=runs[0]['task']=='classification';metric='final_macro_accuracy' if cl else 'final_rmse';avg='average_macro_accuracy' if cl else 'average_rmse'
    best=min(runs,key=lambda s:((-1 if cl else 1)*s[metric],s['config']['ridge']))
    rows.append(dict(dataset=key[0],encoder=key[1],mode=key[2],final=best[metric],average=best[avg],ridge=best['config']['ridge']))
a.out.write_text(json.dumps(rows,indent=2))

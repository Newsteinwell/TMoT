"""Full-versus-projected VAL diagnostic; retain matched-ridge and selected views."""
import argparse,json
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('out',type=Path);a=p.parse_args();groups=defaultdict(list)
terminal=json.loads((a.runs/'complete.json').read_text());assert len(terminal)==144 and all(r['status']=='completed' for r in terminal)
for path in a.runs.glob('*/summary.json'):
    s=json.loads(path.read_text());c=s['config'];groups[c['dataset'],c['encoder'],c['mode']].append(s)
assert len(groups)==24 and all(len(runs)==6 for runs in groups.values())
selected=[];paired=[]
for (d,e,mode),runs in sorted(groups.items()):
    cl=runs[0]['task']=='classification';f='final_macro_accuracy' if cl else 'final_rmse';avg='average_macro_accuracy' if cl else 'average_rmse'
    best=min(runs,key=lambda s:((-1 if cl else 1)*s[f],s['config']['ridge']))
    row=dict(dataset=d,encoder=e,mode=mode,task=best['task'],final=best[f],average=best[avg],ridge=best['config']['ridge'],dimension=best['stages'][-1]['solver']['input_dimension']);selected.append(row);print(json.dumps(row))
    if mode=='full':
        for s in runs:
            old=next(x for x in groups[d,e,'projected'] if x['config']['ridge']==s['config']['ridge']);scale=100 if cl else -1
            paired.append(dict(dataset=d,encoder=e,ridge=s['config']['ridge'],final_gain=(s[f]-old[f])*scale,average_gain=(s[avg]-old[avg])*scale))
a.out.write_text(json.dumps(dict(selected=selected,fixed_ridge_pairs=paired),indent=2))

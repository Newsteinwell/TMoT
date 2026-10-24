"""Equal six-point VAL selection; no TEST reads or dropping failed datasets."""
import argparse,json
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('out',type=Path);a=p.parse_args();groups=defaultdict(list)
terminal=json.loads((a.runs/'complete.json').read_text());assert len(terminal)==108 and all(r['status']=='completed' for r in terminal)
for path in a.runs.glob('*/summary.json'):
    s=json.loads(path.read_text());c=s['config'];groups[c['dataset'],c['encoder'],c['method']].append(s)
assert len(groups)==18 and all(len(v)==6 for v in groups.values())
rows=[]
for (dataset,encoder,method),runs in sorted(groups.items()):
    best=min(runs,key=lambda s:(-s['final_macro_accuracy'],s['config']['ridge']))
    row=dict(dataset=dataset,encoder=encoder,method=method,final=best['final_macro_accuracy'],average=best['average_macro_accuracy'],ridge=best['config']['ridge'],persistent_bytes=best['stages'][-1]['persistent_bytes'],run=Path(best['config']['out']).name)
    rows.append(row);print(json.dumps(row))
a.out.write_text(json.dumps(rows,indent=2))

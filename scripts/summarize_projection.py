"""Best-of-six VAL projection diagnostics, with all raw results retained."""
import argparse,json
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('out',type=Path);a=p.parse_args();groups=defaultdict(list)
for file in a.runs.glob('*/summary.json'):
    s=json.loads(file.read_text());c=s['config'];groups[c['dataset'],c['dimension']].append(s)
assert len(groups)==21 and all(len(v)==6 for v in groups.values())
rows=[]
for (dataset,dimension),runs in sorted(groups.items()):
    classification=runs[0]['task']=='classification';metric='final_macro_accuracy' if classification else 'final_rmse'
    best=min(runs,key=lambda s:((-1 if classification else 1)*s[metric],s['config']['ridge']))
    rows.append(dict(dataset=dataset,dimension=dimension,ridge=best['config']['ridge'],metric=metric,value=best[metric],
                     average=best['average_macro_accuracy' if classification else 'average_rmse'],
                     final_persistent_bytes=best['stages'][-1]['persistent_bytes'],elapsed_s=best['elapsed_s']))
a.out.write_text(json.dumps(dict(scope='same six-value VAL selection per width; random projections are not nested; no TEST',rows=rows),indent=2))
for row in rows:print(json.dumps(row))

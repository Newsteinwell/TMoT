"""Keep all task-loss/replay combinations; equal three-point VAL selection."""
import argparse,json
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('out',type=Path);a=p.parse_args();groups=defaultdict(list)
for path in a.runs.glob('*/summary.json'):
    s=json.loads(path.read_text());c=s['config'];groups[c['dataset'],c['encoder'],c['residual_loss'],c['method']].append(s)
assert len(groups)==36 and all(len(v)==3 for v in groups.values())
rows=[]
for (d,e,loss,mode),runs in sorted(groups.items()):
    classification=runs[0]['task']=='classification';metric='final_macro_accuracy' if classification else 'final_rmse'
    best=min(runs,key=lambda s:((-1 if classification else 1)*s[metric],s['config']['residual_weight']))
    row=dict(dataset=d,encoder=e,loss=loss,mode=mode,metric=metric,final=best[metric],average=best['average_macro_accuracy' if classification else 'average_rmse'],weight=best['config']['residual_weight'])
    rows.append(row);print(json.dumps(row))
a.out.write_text(json.dumps(dict(scope='VAL best of identical three-point regularization grids; one seed/order; no independent test evidence',rows=rows),indent=2))

"""Compare transport to the complete fixed-grid results, not one control ridge."""
import argparse,json
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('fixed',type=Path);p.add_argument('out',type=Path);a=p.parse_args();groups=defaultdict(list)
for folder,mode in ((a.runs,'transport'),(a.fixed,'fixed')):
    for path in folder.glob('*/summary.json'):
        s=json.loads(path.read_text());c=s['config']
        if c['encoder']=='mantis' or (mode=='transport' and c['method']!='transport'):continue
        groups[c['dataset'],c['encoder'],mode].append(s)
assert len(groups)==42 and all(len(v)==6 for v in groups.values())
rows=[]
for (dataset,encoder,mode),runs in sorted(groups.items()):
    classification=runs[0]['task']=='classification';metric='final_macro_accuracy' if classification else 'final_rmse'
    best=min(runs,key=lambda s:((-1 if classification else 1)*s[metric],s['config']['ridge']))
    row=dict(dataset=dataset,encoder=encoder,mode=mode,metric=metric,final=best[metric],average=best['average_macro_accuracy' if classification else 'average_rmse'],ridge=best['config']['ridge'])
    rows.append(row)
a.out.write_text(json.dumps(dict(scope='same six-point VAL ridge selection for both modes; one seed/order; no TEST',rows=rows),indent=2))
for row in rows:print(row['dataset'],row['encoder'],row['mode'],row['final'],row['average'])

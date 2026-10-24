"""Descriptive paired changes; never pool incompatible metric units."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('out',type=Path);a=p.parse_args();rows=[]
for path in sorted(a.runs.glob('*_arrived/summary.json')):
    s=json.loads(path.read_text());first=path.parent.with_name(path.parent.name.removesuffix('_arrived')+'_first')
    f=json.loads((first/'summary.json').read_text());cl=s['task']=='classification'
    final='final_macro_accuracy' if cl else 'final_rmse';avg='average_macro_accuracy' if cl else 'average_rmse';scale=100 if cl else -1
    rows.append(dict(dataset=s['config']['dataset'],mode=s['config']['method'],order=s['config']['order'],first=f[final],arrived=s[final],final_gain=(s[final]-f[final])*scale,average_gain=(s[avg]-f[avg])*scale))
assert len(rows)==20
a.out.write_text(json.dumps(rows,indent=2))

import argparse,json
from pathlib import Path
from collections import defaultdict
p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('previous',type=Path);p.add_argument('out',type=Path);a=p.parse_args();groups=defaultdict(list)
for folder,version in ((a.runs,'regularized'),(a.previous,'joint')):
    for path in folder.glob('*/summary.json'):
        s=json.loads(path.read_text());c=s['config']
        if version=='regularized' and not c['regularized']:continue
        if version=='joint' and not c['completion']:continue
        groups[c['dataset'],c['encoder'],c['method'],version].append(s)
assert len(groups)==36 and all(len(v)==6 for v in groups.values())
rows=[]
for key,runs in sorted(groups.items()):
    cl=runs[0]['task']=='classification';metric='final_macro_accuracy' if cl else 'final_rmse';avg='average_macro_accuracy' if cl else 'average_rmse'
    best=min(runs,key=lambda s:((-1 if cl else 1)*s[metric],s['config']['ridge']))
    rows.append(dict(dataset=key[0],encoder=key[1],method=key[2],version=key[3],final=best[metric],average=best[avg],ridge=best['config']['ridge'],persistent_bytes=best['stages'][-1]['persistent_bytes']))
a.out.write_text(json.dumps(rows,indent=2))
for r in rows:print(r['dataset'],r['encoder'],r['method'],r['version'],round(r['final'],5),round(r['average'],5))

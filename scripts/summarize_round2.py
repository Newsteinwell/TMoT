"""Group by explicit variant name, preserving every observed trajectory."""
import argparse,csv,json,statistics
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True);groups=defaultdict(list);individual=[]
for f in sorted(a.runs.glob('*/summary.json')):
    r=json.loads(f.read_text());name=f.parent.name;dataset=r['config']['dataset'];variant=name[len(dataset)+1:].split('_o')[0]
    r['run_id']=name;r['variant']=variant;groups[(dataset,variant)].append(r)
    individual.append({k:v for k,v in r.items() if k not in ('stages','matrix')})
rows=[]
for (d,v),runs in groups.items():
    metric='final_rmse' if runs[0]['task']=='regression' else 'final_macro_accuracy'
    values=[r[metric] for r in runs]
    rows.append(dict(dataset=d,variant=v,n=len(runs),metric=metric,mean=statistics.mean(values),
                     sd=statistics.stdev(values) if len(values)>1 else 0,
                     elapsed_mean=statistics.mean(r['elapsed_s'] for r in runs)))
with (a.out/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(a.out/'individual_runs.json').write_text(json.dumps(individual,indent=2))
(a.out/'coverage.json').write_text(json.dumps(dict(completed=len(individual),evaluation_splits=sorted(set(r['config']['eval_split'] for r in individual))),indent=2))
print(json.dumps(rows,indent=2))

"""Descriptive paired variability; runs are not independent datasets."""
import argparse,json,math,statistics as st
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
files=list(a.runs.glob('*/summary.json'));assert len(files)==100;lookup={}
for f in files:
 s=json.loads(f.read_text());c=s['config'];key=c['dataset'],c['order'],c['seed'],c['method'];assert key not in lookup;lookup[key]=s

def stats(values):
 assert values and all(math.isfinite(v) for v in values)
 return dict(n=len(values),mean=st.mean(values),sd=st.stdev(values) if len(values)>1 else 0.,median=st.median(values),min=min(values),max=max(values))
rows=[];paired=[]
for dataset in ('har','uwave','dailysports','energy'):
 cl=dataset!='energy';orders=[17,23,29] if cl else [17];metric='final_macro_accuracy' if cl else 'final_rmse';avg='average_macro_accuracy' if cl else 'average_rmse';scale=100 if cl else 1;direction=1 if cl else -1
 pairs=[]
 for order in orders:
  for seed in (42,43,44,45,46):
   local=lookup[dataset,order,seed,'local'];er=lookup[dataset,order,seed,'er']
   r=dict(dataset=dataset,order=order,seed=seed,selection_control=(order==17 and seed==42),local=local[metric]*scale,er=er[metric]*scale,delta=(local[metric]-er[metric])*scale*direction,average_delta=(local[avg]-er[avg])*scale*direction,local_average=local[avg]*scale,er_average=er[avg]*scale)
   pairs.append(r);paired.append(r)
 for name,subset in [('all',pairs),('excluding_selection_control',[r for r in pairs if not r['selection_control']])]+[(f'order_{order}',[r for r in pairs if r['order']==order]) for order in orders]:
  row=dict(dataset=dataset,subset=name,unit='percentage_points' if cl else 'RMSE',**{k:stats([r[k] for r in subset]) for k in ('local','er','delta','average_delta','local_average','er_average')},wins=sum(r['delta']>1e-12 for r in subset),ties=sum(abs(r['delta'])<=1e-12 for r in subset),losses=sum(r['delta']< -1e-12 for r in subset))
  if not cl:row['relative_rmse_gain_percent']=100*row['delta']['mean']/row['er']['mean']
  rows.append(row)
report=dict(scope='Repeated stochastic fits and class orders on previously reused VAL; descriptive variability, not independent datasets or test generalization',positive_delta='favors local',missing=0,rows=rows,paired=paired)
a.out.write_text(json.dumps(report,indent=2))
for r in rows:
 if r['subset']=='all':print(r['dataset'],r['local']['mean'],r['er']['mean'],'paired delta',r['delta']['mean'],'sd',r['delta']['sd'],'win/tie/loss',r['wins'],r['ties'],r['losses'],'avg delta',r['average_delta']['mean'])

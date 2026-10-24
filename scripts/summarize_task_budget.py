import json,sys
from pathlib import Path
root=Path(sys.argv[1]);runs=json.loads((root/'task_budget_v57_runs.json').read_text());prior=json.loads((root/'synthetic_capacity_v54_comparison.json').read_text())['selected'];rows=[]
for d,e in sorted({(s['config']['dataset'],s['config']['encoder']) for s in runs}):
 group=[s for s in runs if s['config']['dataset']==d and s['config']['encoder']==e];cl=group[0]['task']=='classification';metric='macro_accuracy' if cl else 'rmse'
 task=next(s for s in prior if (s['dataset'],s['encoder'],s['variant'])==(d,e,'task_n256'))
 row=dict(dataset=d,encoder=e,task=group[0]['task'],tmot=task)
 for variant in ('er10','moment_er10'):
  s=min((s for s in group if s['config']['variant']==variant),key=lambda s:((-1 if cl else 1)*s['final_'+metric],s['config']['ridge']))
  row[variant]=dict(final=s['final_'+metric],average=s['average_'+metric],ridge=s['config']['ridge'],persistent_bytes=s['stages'][-1]['persistent_bytes'],cache=s['stages'][-1]['buffer_audit']['after'],tmot_final_gain=(task['final']-s['final_'+metric])*(100 if cl else -1),tmot_average_gain=(task['average']-s['average_'+metric])*(100 if cl else -1),tmot_relative_rmse_gain=None if cl else 1-task['final']/s['final_'+metric])
 rows.append(row)
 print(d,e,'task',task['final'],'ER10',row['er10']['final'],'momentER10',row['moment_er10']['final'],flush=True)
(root/'task_budget_v57_comparison.json').write_text(json.dumps(dict(rows=rows,note='Development-only, seed42/order17, best final validation ridge per variant; no TEST; ordinary ER uses standard ridge, moment ER uses regularized innovation; synthetic/moment memory not matched by sample-count constraint.'),indent=2))

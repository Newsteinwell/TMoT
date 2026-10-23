"""Summarize convergence separately from predictive performance on shared VAL."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('old',type=Path);p.add_argument('out',type=Path);a=p.parse_args();rows=[];stages=[]
for path in sorted(a.runs.glob('*/summary.json')):
    s=json.loads(path.read_text());old=json.loads((a.old/path.parent.name/'summary.json').read_text());classification=s['task']=='classification'
    metric='final_macro_accuracy' if classification else 'final_rmse'
    average='average_macro_accuracy' if classification else 'average_rmse'
    diagnostics=[r['evidence']['refinement'] for r in s['stages']];stages.extend(diagnostics)
    rows.append(dict(run=path.parent.name,metric=metric,old_final=old[metric],new_final=s[metric],old_average=old[average],new_average=s[average],
                     total_stages=len(diagnostics),gradient_met=sum(d['gradient_tolerance_met'] for d in diagnostics),optimizer_success=sum(d['optimizer_success'] for d in diagnostics),
                     final_gradient=diagnostics[-1]['normalized_projected_gradient'],final_bound_parameters=diagnostics[-1]['active_bounds']))
assert len(rows)==18
result=dict(scope='same fixed training objective, no VAL-based stopping, not global-optimality evidence',runs=rows,total_stages=len(stages),
            gradient_tolerance_met=sum(d['gradient_tolerance_met'] for d in stages),optimizer_success=sum(d['optimizer_success'] for d in stages))
a.out.write_text(json.dumps(result,indent=2))
print({k:v for k,v in result.items() if k!='runs'})
for row in rows:print(row['run'],row['old_final'],row['new_final'],row['gradient_met'],row['total_stages'],row['final_gradient'])

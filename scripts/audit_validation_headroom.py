"""Development headroom only; never substitutes for sealed TEST criteria."""
import json,collections
from pathlib import Path
root=Path(__file__).resolve().parents[1];groups=collections.defaultdict(list)
new=('natops','epilepsy','racketsports','pendigits','articulary')
for queue in ('round3_validation_v9','nonlinear_validation_v12','fecam_validation_v13'):
    for r in json.loads((root/'results'/queue/'individual_runs.json').read_text()):
        c=r['config'];d=c['dataset']
        if d not in new or c['eval_split']!='val':continue
        if queue=='round3_validation_v9':
            # Only declared baselines, excluding transport candidate.
            if not(c.get('method') in ('ridge','slda') or c.get('no_transport')):continue
        elif queue=='nonlinear_validation_v12' and c.get('method')!='er':continue
        groups[d,queue,r['variant']].append(r['final_macro_accuracy'])
rows=[]
for dataset in new:
    candidates=[(sum(v)/len(v),q,variant) for (d,q,variant),v in groups.items() if d==dataset]
    score,q,variant=max(candidates)
    headroom=100*(1-score)
    rows.append(dict(dataset=dataset,best_declared_baseline_val=score,queue=q,variant=variant,
                     maximum_possible_gain_pp=headroom,one_pp_gain_possible_on_this_val=headroom>=1-1e-12))
report=dict(scope='development VAL only; no inference about untouched TEST headroom',rows=rows,
            eligible_count=sum(r['one_pp_gain_possible_on_this_val'] for r in rows))
(root/'results'/'validation_headroom_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

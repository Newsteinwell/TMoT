import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('results',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
new=[json.loads(f.read_text()) for f in a.runs.glob('*/summary.json')];assert len(new)==56
v7=json.loads((a.results/'stable_development_v7/individual_runs.json').read_text());v11=json.loads((a.results/'nonlinear_matched_v11/individual_runs.json').read_text());rows=[]
for dataset in ('har','uwave','dailysports','energy'):
    cl=dataset!='energy';metric='final_macro_accuracy' if cl else 'final_rmse';avg='average_macro_accuracy' if cl else 'average_rmse'
    groups={}
    for method in ('local','er'):
        groups['v42_'+method]=[s for s in new if s['config']['dataset']==dataset and s['config']['method']==method and s['config']['regularized']]
        groups['joint_control_'+method]=[s for s in new if s['config']['dataset']==dataset and s['config']['method']==method and not s['config']['regularized']]
    groups['v7_ridge']=[s for s in v7 if s['config']['dataset']==dataset and s['config']['method']=='ridge' and s['config']['seed']==42]
    groups['v11_er']=[s for s in v11 if s['config']['dataset']==dataset and s['config']['method']=='er' and s['config']['seed']==42]
    for method,runs in groups.items():
        assert len(runs)==(1 if method.startswith('joint_control') else 3 if method=='v11_er' else 6)
        best=min(runs,key=lambda s:((-1 if cl else 1)*s[metric],s['config']['ridge'],s['config'].get('residual_weight',0)))
        rows.append(dict(dataset=dataset,method=method,final=best[metric],average=best[avg],ridge=best['config']['ridge'],selected_run=Path(best['config']['out']).name,candidates=len(runs),persistent_bytes=best['stages'][-1]['persistent_bytes'] if 'stages' in best else None))
a.out.write_text(json.dumps(rows,indent=2))
for r in rows:print(r['dataset'],r['method'],round(r['final'],5),round(r['average'],5))

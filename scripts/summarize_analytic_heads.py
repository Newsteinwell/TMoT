import json,sys
from pathlib import Path
root=Path(sys.argv[1]);runs=json.loads((root/'analytic_heads_v60_runs.json').read_text());tasks=json.loads((root/'synthetic_capacity_v54_comparison.json').read_text())['selected'];fec=json.loads((root/'fecam_harder_v59_runs.json').read_text());selected=[];comparisons=[]
for d,e in sorted({(r['config']['dataset'],r['config']['encoder']) for r in runs}):
 task=next(r for r in tasks if (r['dataset'],r['encoder'],r['variant'])==(d,e,'task_n256'))
 fe=min((r for r in fec if (r['config']['dataset'],r['config']['encoder'])==(d,e)),key=lambda r:(-r['final_macro_accuracy'],r['config']['alpha1']))
 row=dict(dataset=d,encoder=e,tmot_final=task['final'],tmot_average=task['average'],fecam_final=fe['final_macro_accuracy'],fecam_average=fe['average_macro_accuracy'],heads={})
 for norm in ('raw','unit'):
  for method in ('ncm','slda','ridge'):
   candidates=[r for r in runs if (r['config']['dataset'],r['config']['encoder'],r['config']['normalization'],r['config']['method'])==(d,e,norm,method)]
   def parameter(r):
    name=r['config']['setting'];return 0 if name=='ncm' else float(name.split('_')[1])
   best=min(candidates,key=lambda r:(-r['final_macro_accuracy'],parameter(r)))
   selected.append(best);row['heads'][norm+'_'+method]=dict(final=best['final_macro_accuracy'],average=best['average_macro_accuracy'],setting=best['config']['setting'],tmot_final_gain_pp=100*(task['final']-best['final_macro_accuracy']),tmot_average_gain_pp=100*(task['average']-best['average_macro_accuracy']))
 comparisons.append(row);print(json.dumps(row),flush=True)
(root/'analytic_heads_v60_comparison.json').write_text(json.dumps(dict(rows=comparisons,selected=selected,note='Development only seed42/order17; raw/unit remain separate conditions; no TEST. Fixed final-VAL selection, all settings preserved.'),indent=2))

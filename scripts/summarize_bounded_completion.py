import json,sys
from pathlib import Path
root=Path(sys.argv[1]);runs=json.loads((root/'bounded_completion_v63_runs.json').read_text());old=json.loads((root/'synthetic_capacity_v54_comparison.json').read_text())['selected'];heads=json.loads((root/'analytic_heads_v60_comparison.json').read_text())['rows'];rows=[];selected=[]
for d,e in sorted({(r['config']['dataset'],r['config']['encoder']) for r in runs}):
 t=next(r for r in old if (r['dataset'],r['encoder'],r['variant'])==(d,e,'task_n256'));prior=next(r for r in heads if (r['dataset'],r['encoder'])==(d,e));best=max([('fecam',prior['fecam_final'])]+[(k,v['final']) for k,v in prior['heads'].items()],key=lambda z:z[1]);row=dict(dataset=d,encoder=e,original_tmot_final=t['final'],original_tmot_average=t['average'],strongest_analytic_reference=best[0],strongest_analytic_final=best[1],modes={})
 for mode in ('complete','bounded'):
  s=min((r for r in runs if (r['config']['dataset'],r['config']['encoder'],r['config']['mode'])==(d,e,mode)),key=lambda r:(-r['final_macro_accuracy'],r['config']['alpha']));selected.append(s)
  row['modes'][mode]=dict(final=s['final_macro_accuracy'],average=s['average_macro_accuracy'],alpha=s['config']['alpha'],persistent_arrays_bytes=s['stages'][-1]['persistent_arrays_bytes'])
 chosen_alpha=row['modes']['bounded']['alpha']
 peer=next(r for r in runs if (r['config']['dataset'],r['config']['encoder'],r['config']['mode'],r['config']['alpha'])==(d,e,'complete',chosen_alpha))
 row['complete_at_bounded_alpha']=dict(final=peer['final_macro_accuracy'],average=peer['average_macro_accuracy'],alpha=chosen_alpha)
 rows.append(row);print(json.dumps(row),flush=True)
(root/'bounded_completion_v63_comparison.json').write_text(json.dumps(dict(rows=rows,selected=selected,note='Full class-incremental development trajectories, seed42/order17; no TEST; per-mode final-VAL alpha selection, not robust held-out evidence.'),indent=2))

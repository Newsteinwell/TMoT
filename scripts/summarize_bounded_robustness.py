import json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);runs=json.loads((root/'bounded_robustness_v64_runs.json').read_text());pairs=[];summary=[]
for r in runs:
 c=r['config']
 if c['mode']!='bounded':continue
 for mode in ('complete','original','fecam','ridge'):
  peer=next(p for p in runs if all(p['config'][k]==c[k] for k in ('dataset','encoder','seed','order')) and p['config']['mode']==mode)
  pairs.append(dict(dataset=c['dataset'],encoder=c['encoder'],seed=c['seed'],order=c['order'],reference=mode,development_control=(c['seed'],c['order'])==(42,17),final_gain_pp=100*(r['final_macro_accuracy']-peer['final_macro_accuracy']),average_gain_pp=100*(r['average_macro_accuracy']-peer['average_macro_accuracy']),bounded_final=r['final_macro_accuracy'],reference_final=peer['final_macro_accuracy']))
for d,e,ref in sorted({(p['dataset'],p['encoder'],p['reference']) for p in pairs}):
 rows=[p for p in pairs if (p['dataset'],p['encoder'],p['reference'])==(d,e,ref) and not p['development_control']]
 final=[p['final_gain_pp'] for p in rows];average=[p['average_gain_pp'] for p in rows]
 s=dict(dataset=d,encoder=e,reference=ref,n=len(rows),mean_final_gain_pp=float(np.mean(final)),min_final_gain_pp=min(final),max_final_gain_pp=max(final),mean_average_gain_pp=float(np.mean(average)),final_positive=sum(x>1e-10 for x in final),final_at_least_1pp=sum(x>=1 for x in final),average_positive=sum(x>1e-10 for x in average))
 summary.append(s);print(json.dumps(s),flush=True)
(root/'bounded_robustness_v64_comparison.json').write_text(json.dumps(dict(pairs=pairs,summary=summary,note='Descriptive fixed-parameter repeatability on development TRAIN/VAL, not independent held-out evidence; seed42/order17 controls excluded from summary, all retained. Five seed/order combinations are correlated, not independent datasets.'),indent=2))

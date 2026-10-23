"""Aggregate completed runs only; preserve individual results and missing/failed counts."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=True);groups=defaultdict(list);individual=[]
for file in sorted(a.runs.glob('*/summary.json')):
    r=json.loads(file.read_text());cfg=r['config'];key=(cfg['dataset'],cfg['method'],cfg['encoder'])
    groups[key].append(r)
    individual.append({k:v for k,v in r.items() if k not in ('stages','matrix')})
rows=[]
for (d,m,e),values in groups.items():
    row={'dataset':d,'method':m,'encoder':e,'n_runs':len(values)}
    keys=['final_macro_accuracy','average_macro_accuracy','forgetting'] if values[0]['task']=='classification' else ['final_rmse','average_rmse']
    keys+=['elapsed_s','max_rss_bytes','gpu_peak_bytes']
    for key in keys:
        x=[r[key] for r in values];row[key+'_mean']=float(np.mean(x));row[key+'_sd']=float(np.std(x,ddof=1)) if len(x)>1 else None
    row['persistent_bytes_mean']=float(np.mean([r['stages'][-1]['persistent_bytes'] for r in values]));rows.append(row)
fields=list(dict.fromkeys(k for r in rows for k in r))
with (a.out/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
(a.out/'individual_runs.json').write_text(json.dumps(individual,indent=2))
manifest=json.loads((a.runs/'queue_manifest.json').read_text());expected=len(manifest['jobs'])
(a.out/'coverage.json').write_text(json.dumps({'expected':expected,'completed':len(individual),'missing_or_incomplete':expected-len(individual),'exploratory':True},indent=2))
print(json.dumps(rows,indent=2))

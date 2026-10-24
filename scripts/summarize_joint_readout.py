"""Audit saved final predictions and aggregate final-only fixed-function diagnostics."""
import argparse,hashlib,json,statistics
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();folder=a.root/'joint_readout_v50'
terminal=json.loads((folder/'complete.json').read_text());assert len(terminal)==12 and all(r['status']=='completed' for r in terminal)
rows=[];selected=[]
for path in sorted(folder.glob('*/report.json')):
 r=json.loads(path.read_text());assert r['complete'] and len(r['rows'])==24
 sample=r['rows'][0];d=sample['dataset'];e=sample['encoder']
 for split,h in r['input_hashes'].items():
  with (a.root/'features'/e/f'{d}_{split}.npz').open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==h
 for row in r['rows']:
  with np.load(path.parent/f"{row['variant']}_r{row['ridge']:g}.npz") as p:
   for key,field in [('pred','final_exact'),('old_pred','final_original'),('joint_approx_pred','final_joint_approx'),('joint_exact_pred','final_joint_exact')]:
    y=p['y'];pred=p[key];score=float(np.mean([np.mean(pred[y==c]==y[y==c]) for c in np.unique(y)])) if row['task']=='classification' else float(np.sqrt(np.mean((pred-y)**2)))
    np.testing.assert_allclose(score,row[field],rtol=1e-12,atol=1e-12)
  rows.append(row)
 for variant in sorted({row['variant'] for row in r['rows']}):
  group=[x for x in r['rows'] if x['variant']==variant];sign=-1 if group[0]['task']=='classification' else 1
  old=min(group,key=lambda x:(sign*x['final_original'],x['ridge']));exact=min(group,key=lambda x:(sign*x['final_exact'],x['ridge']))
  chosen={field:min(group,key=lambda x:(sign*x[field],x['ridge'])) for field in ['final_original','final_exact','final_joint_approx','final_joint_exact']}
  selected.append(dict(dataset=d,encoder=e,variant=variant,task=old['task'],values={field:row[field] for field,row in chosen.items()},ridges={field:row['ridge'] for field,row in chosen.items()}))
assert len(rows)==288 and len(selected)==48
report=dict(complete=True,rows=rows,selected=selected,max_independent_readout_error=max(x['independent_readout_max_error'] for x in rows),note='Final-only full-history diagnostic, not a continual learner')
a.out.write_text(json.dumps(report,indent=2))
for row in selected:
 if row['variant']=='rank16_shrink01':print(row)
print('median hidden-target error',statistics.median(x['hidden_target_relative'] for x in rows if x['ridge']==.1))

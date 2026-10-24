import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();folder=a.root/'runs/synthetic_robustness_v55'
terminal=json.loads((folder/'complete.json').read_text());assert len(terminal)==120 and all(x['status']=='completed' for x in terminal)
runs=[json.loads(p.read_text()) for p in folder.glob('*/summary.json')];assert len(runs)==120
by={ (x['config']['dataset'],x['config']['encoder'],x['config']['seed'],x['config']['order'],x['config']['variant']):x for x in runs};pairs=[]
for key,s in by.items():
 d,e,seed,order,v=key
 if v=='task':continue
 t=by[d,e,seed,order,'task'];cl=s['task']=='classification';f='final_macro_accuracy' if cl else 'final_rmse';av='average_macro_accuracy' if cl else 'average_rmse';scale=100 if cl else -1
 pairs.append(dict(dataset=d,encoder=e,seed=seed,order=order,reference=v,task=s['task'],development_control=seed==42 and order==17,task_final=t[f],reference_final=s[f],final_gain=(t[f]-s[f])*scale,average_gain=(t[av]-s[av])*scale,rmse_relative_gain=None if cl else 1-t[f]/s[f],task_bytes=t['stages'][-1]['persistent_bytes'],reference_bytes=s['stages'][-1]['persistent_bytes']))
summary=[]
for d,e,ref in sorted({(r['dataset'],r['encoder'],r['reference']) for r in pairs}):
 for new_only in (False,True):
  group=[r for r in pairs if (r['dataset'],r['encoder'],r['reference'])==(d,e,ref) and (not new_only or not r['development_control'])]
  summary.append(dict(dataset=d,encoder=e,reference=ref,new_combinations_only=new_only,n=len(group),mean_final_gain=statistics.mean(r['final_gain'] for r in group),median_final_gain=statistics.median(r['final_gain'] for r in group),mean_average_gain=statistics.mean(r['average_gain'] for r in group),wins=sum(r['final_gain']>1e-10 for r in group),ties=sum(abs(r['final_gain'])<=1e-10 for r in group),losses=sum(r['final_gain']< -1e-10 for r in group),clear_gain_streams=sum(r['final_gain']>=1 if r['task']=='classification' else r['rmse_relative_gain']>=.05 for r in group)))
a.out.write_text(json.dumps(dict(complete=True,pairs=pairs,summary=summary,note='Descriptive paired development results; repeated seeds/orders are not independent datasets; no TEST evaluated'),indent=2))
for row in summary:
 if row['new_combinations_only']:print(row,flush=True)

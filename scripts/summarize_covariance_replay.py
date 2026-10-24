"""Matched-grid sampler comparison with predeclared same-pipeline ER references."""
import argparse,json
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('mirror',type=Path);p.add_argument('out',type=Path);a=p.parse_args();groups=defaultdict(list);all_current=[]
for path in (a.mirror/'covariance_replay_v48').glob('*/summary.json'):
    s=json.loads(path.read_text());c=s['config'];groups[c['dataset'],c['encoder'],c['variant']].append((s,path));all_current.append(s)
assert len(all_current)==288 and len(groups)==48
for queue in ('harder_validation_v45','regularized_innovation_v41','history_scale_v42','latent_replay_v47'):
    for path in (a.mirror/queue).glob('*/summary.json'):
        s=json.loads(path.read_text());c=s['config'];d=c['dataset'];e=c['encoder']
        if c['method']!='er' or not c.get('regularized',True):continue
        allowed=(queue=='harder_validation_v45' and d in ('phonemespectra','lsst','handwriting')) or (queue=='regularized_innovation_v41' and d in ('beijing_pm25','air_quality')) or (queue=='history_scale_v42' and d=='energy' and e=='mantis') or (queue=='latent_replay_v47' and d=='energy' and e=='summary')
        if allowed:groups[d,e,'er'].append((s,path))
assert len(groups)==60 and all(len(v)==6 for v in groups.values()),{k:len(v) for k,v in groups.items()}
selected=[];paired=[]
for (d,e,variant),runs in sorted(groups.items()):
    cl=runs[0][0]['task']=='classification';final='final_macro_accuracy' if cl else 'final_rmse';avg='average_macro_accuracy' if cl else 'average_rmse';scale=100 if cl else -1
    best,path=min(runs,key=lambda sp:((-1 if cl else 1)*sp[0][final],sp[0]['config']['ridge']))
    row=dict(dataset=d,encoder=e,variant=variant,task=best['task'],final=best[final],average=best[avg],ridge=best['config']['ridge'],source_queue=path.parent.parent.name,source_run=path.parent.name,persistent_bytes=best['stages'][-1]['persistent_bytes']);selected.append(row)
    if variant!='er':
        for other,factor in [('rank16_shrink01','vs_control')]+([('rank16_shrink0','rank_at_shrink0')] if variant=='full_shrink0' else [])+([('full_shrink01','shrink_at_full')] if variant=='full_shrink0' else []):
            for s,path in runs:
                old=next(t for t,p in groups[d,e,other] if t['config']['ridge']==s['config']['ridge'])
                paired.append(dict(dataset=d,encoder=e,variant=variant,reference=other,factor=factor,ridge=s['config']['ridge'],final_gain=(s[final]-old[final])*scale,average_gain=(s[avg]-old[avg])*scale))
a.out.write_text(json.dumps(dict(selected=selected,fixed_ridge_pairs=paired),indent=2))
for d,e in sorted({(r['dataset'],r['encoder']) for r in selected}):
    rows={r['variant']:r for r in selected if r['dataset']==d and r['encoder']==e};scale=100 if rows['er']['task']=='classification' else 1
    print(d,e,'rank16_shrink01/rank16_shrink0/full_shrink01/full_shrink0/ER',*[round(rows[v]['final']*scale,5) for v in ('rank16_shrink01','rank16_shrink0','full_shrink01','full_shrink0','er')])

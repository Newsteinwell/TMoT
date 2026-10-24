"""Remove unselected completed HPO checkpoints, preserving all evidence files."""
import argparse, hashlib, json, os
from collections import defaultdict
from pathlib import Path
QUEUES = ('stable_development_v7','kernel_development_v8','round3_validation_v9','nonlinear_development_v10','nonlinear_matched_v11','nonlinear_validation_v12','local_development_v14','target_development_v15','calibrated_development_v16','causal_development_v19_retry','task_loss_v26','raw_transport_v29','local_warm_v32','local_support_v33','local_merge_v34')
def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('manifest',type=Path);p.add_argument('--apply',action='store_true');a=p.parse_args()
root=a.root.resolve(); assert root.name=='tmot-artifacts'; assert not a.manifest.exists()
items=[];kept=[];counts={};evidence={}
for queue in QUEUES:
    folder=root/'runs'/queue
    terminal=json.loads((folder/'complete.json').read_text())
    assert terminal and all(r['status']=='completed' for r in terminal), queue
    paths=sorted(folder.glob('*/summary.json'));assert len(paths)==len(terminal),(queue,len(paths),len(terminal))
    groups=defaultdict(list)
    for path in paths:
        s=json.loads(path.read_text());c=s['config']; assert s['status']=='completed' and c['eval_split']=='val'
        # Vary only search hyperparameters; preserve method, ablation, seed and order.
        key=json.dumps({k:v for k,v in c.items() if k not in ('root','out','ridge','residual_weight','threads','device')},sort_keys=True)
        groups[key].append((path,s))
    before=len(items)
    for rows in groups.values():
        cl=rows[0][1]['task']=='classification'
        metrics=('final_macro_accuracy','average_macro_accuracy') if cl else ('final_rmse','average_rmse')
        selected={min(rows,key=lambda r:((-1 if cl else 1)*r[1][metric],str(r[0])))[0] for metric in metrics}
        # Preserve linear-state controls used by later audits.
        if queue in ('stable_development_v7','round3_validation_v9'):
            selected.update(path for path,s in rows if s['config'].get('ridge')==.1)
        if queue in ('nonlinear_matched_v11','nonlinear_validation_v12'):
            selected.update(path for path,s in rows if s['config'].get('method')=='er' and s['config'].get('residual_weight')==1.)
        # Existing v26/v31 comparisons use v12 weight .1 anchors.
        if queue=='nonlinear_validation_v12':selected.update(path for path,s in rows if s['config'].get('residual_weight')==.1)
        for path,s in rows:
            model=path.parent/'final.pt';assert model.is_file() and not model.is_symlink()
            if path in selected:kept.append(str(model));continue
            st=model.stat();items.append(dict(path=str(model),sha256=digest(model),size=st.st_size,inode=st.st_ino,links=st.st_nlink,source=s['git_commit'],config=s['config']))
    counts[queue]=len(items)-before
    # Prove every non-checkpoint artifact is unchanged after deletion.
    for f in folder.rglob('*'):
        if f.is_file() and f.name!='final.pt': evidence[str(f)]=digest(f)
report=dict(policy='Only listed completed validation HPO queues; retain best final and average per otherwise-identical config plus v7/v9 ridge .1, v11/v12 ER weight 1 and v12 weight .1 anchors. Keep all predictions, summaries, logs and all unlisted queues. Deleted checkpoints require regeneration.',applied=False,kept=kept,removed_candidates=items,counts=counts,evidence_sha256=evidence)
a.manifest.write_text(json.dumps(report,indent=2))
print(json.dumps(dict(candidates=len(items),kept=len(kept),upper_bound_gib=sum(x['size'] for x in items)/1024**3,counts=counts)),flush=True)
if a.apply:
    with a.manifest.with_suffix('.journal.jsonl').open('x') as log:
        for item in items:
            path=Path(item['path']);assert path.is_relative_to(root/'runs') and path.name=='final.pt'
            st=path.stat();assert st.st_ino==item['inode'] and st.st_size==item['size'] and digest(path)==item['sha256']
            log.write(json.dumps(dict(path=str(path),action='unlink_intent',sha256=item['sha256']))+'\n');log.flush();os.fsync(log.fileno())
            path.unlink();log.write(json.dumps(dict(path=str(path),action='unlinked'))+'\n');log.flush()
    assert all(Path(f).is_file() and digest(Path(f))==h for f,h in evidence.items())
    assert all(Path(f).is_file() for f in kept)
    report['applied']=True;report['evidence_verified']=len(evidence);a.manifest.write_text(json.dumps(report,indent=2))
    print('Deletion complete; all non-checkpoint evidence unchanged.',flush=True)

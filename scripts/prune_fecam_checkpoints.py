"""Prune only non-selected, reproducible v13 FeCAM checkpoints; retain evidence."""
import argparse,hashlib,json,os
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('manifest',type=Path);p.add_argument('--apply',action='store_true');a=p.parse_args()
root=a.root.resolve();items=[];kept=[]
for queue in ('fecam_validation_v13',):
    folder=root/'runs'/queue
    terminal=json.loads((folder/'complete.json').read_text())
    assert len(terminal)=={'fecam_validation_v13':96}[queue] and all(r['status']=='completed' for r in terminal)
    groups=defaultdict(list)
    for path in folder.glob('*/summary.json'):
        s=json.loads(path.read_text());c=s['config'];assert c['eval_split']=='val'
        groups[c['dataset'],c['encoder'],c['dimension'],c['seed'],c['order']].append((path,s))
    for rows in groups.values():
        cl=rows[0][1]['task']=='classification';metrics=('final_macro_accuracy','average_macro_accuracy') if cl else ('final_rmse','average_rmse')
        selected={min(rows,key=lambda r:((-1 if cl else 1)*r[1][metric],r[1]['config']['alpha1']))[0] for metric in metrics}
        selected.update(path for path,s in rows if s['config']['alpha1']==1.)
        for path,s in rows:
            model=path.parent/'final.pt'
            if path in selected:kept.append(str(model));continue
            assert model.is_file() and not model.is_symlink()
            stat=model.stat();digest=hashlib.sha256(model.read_bytes()).hexdigest()
            items.append(dict(path=str(model),sha256=digest,size=stat.st_size,inode=stat.st_ino,links=stat.st_nlink,source=s['git_commit'],config=s['config']))
report=dict(policy='v13 FeCAM only; preserve best final and best average checkpoint per dataset/encoder/dimension/seed/order/plus alpha1=1 controls; all predictions, summaries and logs retained; removed checkpoints require regeneration',kept=kept,removed_candidates=items,applied=False)
assert not a.manifest.exists();a.manifest.write_text(json.dumps(report,indent=2))
print(json.dumps(dict(candidates=len(items),kept=len(kept),upper_bound_gib=sum(x['size'] for x in items)/1024**3)))
if a.apply:
    journal=a.manifest.with_suffix('.journal.jsonl')
    with journal.open('x') as log:
        for item in items:
            path=Path(item['path']);assert path.is_relative_to(root/'runs') and path.name=='final.pt'
            stat=path.stat();assert stat.st_ino==item['inode'] and stat.st_size==item['size']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
            log.write(json.dumps(dict(path=str(path),action='unlink_intent',sha256=item['sha256']))+'\n');log.flush();os.fsync(log.fileno())
            path.unlink()
            log.write(json.dumps(dict(path=str(path),action='unlinked'))+'\n');log.flush()
    report['applied']=True;a.manifest.write_text(json.dumps(report,indent=2))
    print('Pruned listed checkpoints; evidence and selected models retained')

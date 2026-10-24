"""Prune superseded v21-v44 checkpoints while preserving best models and evidence."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

p=argparse.ArgumentParser()
p.add_argument('root',type=Path)
p.add_argument('report',type=Path)
p.add_argument('--apply',action='store_true')
a=p.parse_args()
root=a.root.resolve()
assert root==Path('/root/autodl-tmp/tmot-artifacts')
assert not a.report.exists()
groups=defaultdict(list); evidence={}; referenced=set(); skipped=[]
queues=[]
for q in sorted((root/'runs').iterdir()):
    m=re.search(r'_v(\d+)(?:_|$)',q.name)
    if q.is_dir() and m and 21<=int(m[1])<=44:
        queues.append(q)
# Explicit checkpoint paths in newer evidence protect dependencies.
for q in (root/'runs').iterdir():
    m=re.search(r'_v(\d+)(?:_|$)',q.name)
    if not q.is_dir() or not m or int(m[1])<45:continue
    for f in q.rglob('*'):
        if f.is_file() and f.suffix in ('.json','.jsonl'):
            for match in re.findall(r'/root/autodl-tmp/tmot-artifacts/runs/[^\s"\']+?\.pt(?:\.gz)?',f.read_text(errors='replace')):
                referenced.add(match.removesuffix('.gz'))
for q in queues:
    for f in q.rglob('*'):
        if not f.is_file() or f.is_symlink():continue
        if f.name not in ('final.pt','final.pt.gz'):
            if not f.name.endswith(('.pt','.pt.gz')):evidence[str(f)]=sha(f)
            continue
        s=f.parent/'summary.json'
        if not s.is_file():skipped.append(str(f));continue
        v=json.loads(s.read_text());c=v.get('config',{})
        if v.get('status')!='completed' or c.get('eval_split')!='val':
            skipped.append(str(f));continue
        task=v.get('task')
        metrics=('final_macro_accuracy','average_macro_accuracy') if task=='classification' else ('final_rmse','average_rmse')
        if not all(k in v for k in metrics):skipped.append(str(f));continue
        # Keep both objectives for every seed/order, dataset/encoder and method.
        key=(q.name,task,*(str(c.get(k,'')) for k in ('dataset','encoder','method','seed','order')))
        groups[key].append((f,v,metrics))
removed=[]; kept=[]
for rows in groups.values():
    sign=-1 if rows[0][1]['task']=='classification' else 1
    best={min(rows,key=lambda r:(sign*r[1][metric],str(r[0])))[0] for metric in rows[0][2]}
    for f,s,_ in rows:
        if f in best or str(f).removesuffix('.gz') in referenced:
            kept.append(str(f));continue
        st=f.stat()
        if st.st_nlink!=1:skipped.append(str(f));continue
        removed.append(dict(path=str(f),size=st.st_size,sha256=sha(f),inode=st.st_ino,mtime_ns=st.st_mtime_ns,config=s['config'],source=s.get('git_commit')))
report=dict(policy='Superseded v21-v44 completed VAL exploration only. Preserve best final and average per queue/task/dataset/encoder/method/seed/order, explicit checkpoint references from v45+ evidence, all multiply linked files, all other queues and all non-model evidence. Deleted models require regeneration; they are not cold archived.',removed=removed,kept=kept,skipped=skipped,evidence_sha256=evidence,free_before=shutil.disk_usage(root).free,applied=False)
a.report.write_text(json.dumps(report))
print(json.dumps(dict(candidates=len(removed),kept=len(kept),GiB=sum(x['size'] for x in removed)/2**30)),flush=True)
if a.apply:
    # Refuse to proceed if a training process appeared during inventory.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        assert not ('TMoT' in cmd and any(x in cmd for x in ('tmot.experiment','scripts/run_'))),cmd
    with a.report.with_suffix('.journal.jsonl').open('x') as log:
        for item in removed:
            f=Path(item['path']);st=f.stat()
            assert st.st_ino==item['inode'] and st.st_mtime_ns==item['mtime_ns'] and st.st_size==item['size']
            assert sha(f)==item['sha256']
            log.write(json.dumps(item)+'\n');log.flush();os.fsync(log.fileno())
            f.unlink()
    assert all(Path(f).is_file() and sha(Path(f))==h for f,h in evidence.items())
    assert all(Path(f).is_file() for f in kept)
    report.update(applied=True,free_after=shutil.disk_usage(root).free,evidence_verified=len(evidence))
    a.report.write_text(json.dumps(report))
    print(json.dumps({k:report[k] for k in ('applied','free_before','free_after','evidence_verified')}),flush=True)

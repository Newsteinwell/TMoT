"""Auditable disk maintenance: v45 HPO pruning and lossless old-state archiving."""
import argparse, concurrent.futures, gzip, hashlib, json, os, re, shutil
from collections import defaultdict
from pathlib import Path

def digest(p, compressed=False):
    h=hashlib.sha256()
    with (gzip.open(p,'rb') if compressed else p.open('rb')) as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()

def archive(p):
    st=p.stat(); dest=p.with_suffix('.pt.gz'); tmp=dest.with_suffix('.gz.tmp')
    assert not dest.exists() and not tmp.exists() and not p.is_symlink()
    sha=digest(p)
    with p.open('rb') as src, tmp.open('xb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,compresslevel=1,mtime=0) as out:
            shutil.copyfileobj(src,out,8*1024**2)
        raw.flush();os.fsync(raw.fileno())
    assert digest(tmp,True)==sha
    if tmp.stat().st_size>=st.st_size*.95:
        tmp.unlink();return dict(path=str(p),action='kept_low_saving')
    assert p.stat().st_mtime_ns==st.st_mtime_ns and digest(p)==sha
    tmp.rename(dest);p.unlink()
    return dict(path=str(p),archive=str(dest),sha256=sha,original_bytes=st.st_size,archive_bytes=dest.stat().st_size,action='archived_verified')

p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('report',type=Path);a=p.parse_args()
r=a.root.resolve();assert r.name=='tmot-artifacts';assert not a.report.exists()
runs=r/'runs';before=shutil.disk_usage(r)._asdict()
evidence={str(f):digest(f) for f in runs.rglob('*') if f.is_file() and f.suffix not in ('.pt','.gz')}
groups=defaultdict(list);q=runs/'harder_validation_v45'
terminal=json.loads((q/'complete.json').read_text());assert len(terminal)==108 and all(x['status']=='completed' for x in terminal)
for path in q.glob('*/summary.json'):
    s=json.loads(path.read_text());assert s['status']=='completed' and s['config']['eval_split']=='val'
    key=json.dumps({k:v for k,v in s['config'].items() if k not in ('ridge','root','out','device','threads')},sort_keys=True)
    groups[key].append((path,s))
remove=[];keep=[]
for rows in groups.values():
    selected={min(rows,key=lambda x:(-x[1][m],str(x[0])))[0] for m in ('final_macro_accuracy','average_macro_accuracy')}
    selected.update(path for path,s in rows if s['config']['ridge']==.1)
    for path,s in rows:
        f=path.parent/'final.pt';assert f.is_file()
        if path in selected:keep.append(str(f))
        else:remove.append(dict(path=str(f),size=f.stat().st_size,sha256=digest(f),config=s['config'],source=s['git_commit']))
candidates=[]
for folder in runs.iterdir():
    match=re.search(r'_v(\d+)$',folder.name)
    if not match or int(match[1])>30:continue
    for f in folder.glob('*/final.pt'):
        summary=f.parent/'summary.json'
        if f.stat().st_nlink==1 and summary.exists() and json.loads(summary.read_text()).get('status')=='completed':candidates.append(f)
report=dict(before=before,removed=remove,kept_v45=keep,archive_candidates=[str(f) for f in candidates],evidence_sha256=evidence,complete=False)
a.report.write_text(json.dumps(report,indent=2))
with a.report.with_suffix('.journal.jsonl').open('x') as log:
    def record(item):
        log.write(json.dumps(item)+'\n');log.flush();os.fsync(log.fileno())
    for item in remove:
        f=Path(item['path']);assert digest(f)==item['sha256'];record(dict(action='unlink_intent',**item));f.unlink();record(dict(action='unlinked',path=str(f)))
    print(json.dumps(dict(pruned=len(remove),pruned_gib=sum(x['size'] for x in remove)/2**30,archive_candidates=len(candidates))),flush=True)
    archived=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i,item in enumerate(pool.map(archive,candidates)):
            archived.append(item);record(item)
            if (i+1)%50==0:print('archived/checked',i+1,flush=True)
assert all(Path(f).exists() and digest(Path(f))==h for f,h in evidence.items())
assert all(Path(f).exists() for f in keep)
report.update(complete=True,archives=archived,evidence_verified=len(evidence),after=shutil.disk_usage(r)._asdict())
a.report.write_text(json.dumps(report,indent=2));print(json.dumps(dict(complete=True,before=before,after=report['after'],evidence_verified=len(evidence))),flush=True)

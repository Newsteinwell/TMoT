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

p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('report',type=Path);p.add_argument('--queues',nargs='+',default=['local_warm_v32','local_support_v33','local_merge_v34']);a=p.parse_args();r=a.root.resolve();assert r.name=='tmot-artifacts' and not a.report.exists()
queues=a.queues;assert all(re.fullmatch(r'[a-z0-9_]+',q) for q in queues);candidates=[];evidence={}
for queue in queues:
 folder=r/'runs'/queue
 for file in folder.rglob('*'):
  if file.is_file() and file.suffix not in ('.pt','.gz'):evidence[str(file)]=digest(file)
 for file in folder.glob('*/final.pt'):
  assert json.loads((file.parent/'summary.json').read_text())['status']=='completed'
  if file.stat().st_nlink==1:candidates.append(file)
report=dict(complete=False,before=shutil.disk_usage(r)._asdict(),archive_candidates=[str(p) for p in candidates],evidence_sha256=evidence,policy='Lossless only; completed models in explicitly selected queues.',queues=queues)
a.report.write_text(json.dumps(report,indent=2));archives=[]
with a.report.with_suffix('.journal.jsonl').open('x') as log:
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for item in pool.map(archive,candidates):
   archives.append(item);log.write(json.dumps(item)+'\n');log.flush();os.fsync(log.fileno())
assert all(Path(p).is_file() and digest(Path(p))==h for p,h in evidence.items())
report.update(complete=True,archives=archives,after=shutil.disk_usage(r)._asdict(),evidence_verified=len(evidence));a.report.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k in ('complete','before','after','evidence_verified')}))

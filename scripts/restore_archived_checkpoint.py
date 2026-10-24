"""Restore one checkpoint archived by compact_disk, verifying its original hash."""
import argparse, gzip, hashlib, json, os, shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('manifest',type=Path);a=p.parse_args()
report=json.loads(a.manifest.read_text());assert report['complete']
path=a.checkpoint.resolve();entry=next(x for x in report['archives'] if x['path']==str(path) and x['action']=='archived_verified')
archive=Path(entry['archive']);assert not path.exists();tmp=path.with_suffix('.restore.tmp');assert not tmp.exists()
assert shutil.disk_usage(path.parent).free>entry['original_bytes']+1024**3
with gzip.open(archive,'rb') as src,tmp.open('xb') as dst:
    shutil.copyfileobj(src,dst,8*1024**2);dst.flush();os.fsync(dst.fileno())
with tmp.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==entry['sha256']
tmp.rename(path)
print('Restored and SHA256 verified:',path)
# Keep compressed backup until the restored state has been consumed by its audit.

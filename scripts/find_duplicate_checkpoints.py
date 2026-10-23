"""Read-only SHA256 duplicate scan of terminal experiment queues only."""
import argparse,collections,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();sizes=collections.defaultdict(list)
for complete in (a.root/'runs').glob('*/complete.json'):
    for summary in complete.parent.glob('*/summary.json'):
        path=summary.parent/'final.pt'
        if path.is_file() and not path.is_symlink():sizes[path.stat().st_size].append(path)
groups=collections.defaultdict(list);read_bytes=0
for size,paths in sizes.items():
    if len(paths)<2:continue
    for path in sorted(paths):
        before=path.stat();h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
        after=path.stat();assert (before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
        groups[size,h.hexdigest()].append(dict(path=str(path),inode=after.st_ino,device=after.st_dev,mtime_ns=after.st_mtime_ns))
        read_bytes+=size
rows=[]
for (size,digest),files in groups.items():
    inodes={(f['device'],f['inode']) for f in files}
    if len(inodes)>1:rows.append(dict(size=size,sha256=digest,files=files,reclaimable_bytes=(len(inodes)-1)*size))
report=dict(scope='terminal queues, completed trajectories, regular final.pt files only',scanned_bytes=read_bytes,duplicate_groups=rows,reclaimable_bytes=sum(r['reclaimable_bytes'] for r in rows))
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2));print(json.dumps(dict(groups=len(rows),reclaimable_gib=report['reclaimable_bytes']/1024**3,scanned_gib=read_bytes/1024**3)))

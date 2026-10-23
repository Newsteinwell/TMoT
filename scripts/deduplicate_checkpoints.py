"""Atomically hard-link verified identical immutable completed checkpoints."""
import argparse,hashlib,json,os,uuid
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def apply(report, root):
    base=(root/'runs').resolve();changed=[];reclaimed=0
    for group in report['duplicate_groups']:
        paths=[Path(f['path']) for f in group['files']]
        for path in paths:
            if path.is_symlink() or not path.is_file() or path.name!='final.pt' or not path.resolve().is_relative_to(base):
                raise ValueError(f'Invalid checkpoint path: {path}')
            if not (path.parent/'summary.json').is_file() or not (path.parent.parent/'complete.json').is_file():
                raise ValueError('Not a completed trajectory in a terminal queue')
        source=paths[0]
        if source.stat().st_size!=group['size'] or digest(source)!=group['sha256']:raise ValueError('Source changed')
        for target in paths[1:]:
            if os.path.samefile(source,target):continue
            before=target.stat()
            if before.st_dev!=source.stat().st_dev or before.st_size!=group['size'] or digest(target)!=group['sha256']:
                raise ValueError('Target changed or filesystem differs')
            temp=target.with_name('.dedup-'+uuid.uuid4().hex)
            try:
                os.link(source,temp);os.replace(temp,target)
            finally:
                if temp.exists():temp.unlink()
            assert os.path.samefile(source,target) and target.stat().st_size==group['size']
            if before.st_nlink==1:reclaimed+=before.st_size
            changed.append(dict(path=str(target),source=str(source),sha256=group['sha256']))
    return dict(changed=changed,reclaimed_bytes=reclaimed,semantics='hard links of byte-identical immutable checkpoints; all original paths retained')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    result=apply(json.loads(a.manifest.read_text()),a.root);a.out.write_text(json.dumps(result,indent=2));print(json.dumps(dict(linked=len(result['changed']),reclaimed_gib=result['reclaimed_bytes']/1024**3)))

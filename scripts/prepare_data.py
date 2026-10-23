"""Prepare documented pilot splits. Downloads are separate and never committed."""
import argparse, hashlib, io, json, re, zipfile
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(out,name,data,meta):
    out.mkdir(parents=True,exist_ok=True)
    for split in ('train','val','test'):
        x,y,ids,groups=data[split]
        assert len(x)==len(y)==len(ids)==len(groups)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        np.savez_compressed(out/f'{name}_{split}.npz',x=x.astype('float32'),y=y,ids=np.array(ids),groups=np.array(groups))
    ids=[set(data[s][2]) for s in ('train','val','test')]
    assert not(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
    meta['sizes']={s:len(data[s][0]) for s in data}
    meta['shapes']={s:list(data[s][0].shape) for s in data}
    meta['files']={s:sha(out/f'{name}_{s}.npz') for s in data}
    (out/f'{name}_manifest.json').write_text(json.dumps(meta,indent=2))
    print(name,json.dumps(meta['sizes']),flush=True)


def har(raw,out):
    z=zipfile.ZipFile(raw/'har.zip')
    nested=[n for n in z.namelist() if n.endswith('.zip')]
    if nested:z=zipfile.ZipFile(io.BytesIO(z.read(nested[0])))
    prefix=next(n for n in z.namelist() if n.endswith('train/y_train.txt'))[:-len('train/y_train.txt')]
    data={}
    for split in ('train','test'):
        base=prefix+split+'/'
        signals=['body_acc_'+a for a in 'xyz']+['body_gyro_'+a for a in 'xyz']+['total_acc_'+a for a in 'xyz']
        x=np.stack([np.loadtxt(io.BytesIO(z.read(base+'Inertial Signals/'+v+'_'+split+'.txt'))) for v in signals],axis=1)
        y=np.loadtxt(io.BytesIO(z.read(base+'y_'+split+'.txt')),dtype=int)-1
        groups=np.loadtxt(io.BytesIO(z.read(base+'subject_'+split+'.txt')),dtype=int)
        ids=np.array([f'{split}:{j}' for j in range(len(y))])
        if split=='train':
            valgroups=np.unique(groups)[-4:];mask=np.isin(groups,valgroups)
            data['train']=(x[~mask],y[~mask],ids[~mask],groups[~mask]);data['val']=(x[mask],y[mask],ids[mask],groups[mask])
        else:data[split]=(x,y,ids,groups)
    assert not(set(data['train'][3])&set(data['test'][3]))
    write(out,'har',data,{'task':'classification','source':'UCI 240','source_sha256':sha(raw/'har.zip'),
      'protocol':'official subject-disjoint test; largest four training subject IDs for validation',
      'notes':'Existing 50%-overlapping windows; all train/val/test subjects disjoint. Pilot, not TSCIL exact split.'})


def read_ts(content):
    x=[];y=[];started=False
    for line in content.decode('utf-8-sig').splitlines():
        if not started:
            if line.strip().lower()=='@data':started=True
            continue
        if not line.strip() or line.startswith('#'):continue
        parts=line.strip().split(':');x.append(np.array([[float(v) for v in c.split(',')] for c in parts[:-1]]));y.append(parts[-1])
    return np.stack(x),np.array(y)


def uwave(raw,out):
    z=zipfile.ZipFile(raw/'uwave.zip');data={};mapping=None
    for split in ('TRAIN','TEST'):
        path=next(n for n in z.namelist() if n.endswith('_'+split+'.ts'))
        x,ys=read_ts(z.read(path))
        if mapping is None:mapping={c:i for i,c in enumerate(sorted(set(ys)))}
        y=np.array([mapping[c] for c in ys]);ids=np.array([f'{split}:{i}' for i in range(len(y))]);groups=np.full(len(y),-1)
        if split=='TRAIN':
            rng=np.random.default_rng(1701);valid=[]
            for c in np.unique(y):
                rows=np.flatnonzero(y==c);valid.extend(rng.permutation(rows)[:max(1,len(rows)//5)])
            mask=np.isin(np.arange(len(y)),valid)
            data['train']=(x[~mask],y[~mask],ids[~mask],groups[~mask]);data['val']=(x[mask],y[mask],ids[mask],groups[mask])
        else:data['test']=(x,y,ids,groups)
    write(out,'uwave',data,{'task':'classification','source':'UEA multivariate UWaveGestureLibrary',
      'source_sha256':sha(raw/'uwave.zip'),'protocol':'official train/test; stratified 20% train validation',
      'notes':'Subject metadata unavailable in ts files; do not claim subject-independent validation; pilot only.'})


def sports(raw,out):
    z=zipfile.ZipFile(raw/'dailysports.zip')
    nested=[n for n in z.namelist() if n.endswith('.zip')]
    if nested:z=zipfile.ZipFile(io.BytesIO(z.read(nested[0])))
    rows={s:[[],[],[],[]] for s in ('train','val','test')}
    for name in sorted(z.namelist()):
        m=re.search(r'a(\d+)/p(\d+)/s(\d+)\.txt$',name)
        if m is None:continue
        c,p,seg=map(int,m.groups());split='train' if p<=5 else 'val' if p==6 else 'test'
        rows[split][0].append(np.loadtxt(io.BytesIO(z.read(name)),delimiter=',').T)
        rows[split][1].append(c-1);rows[split][2].append(f'a{c}:p{p}:s{seg}');rows[split][3].append(p)
    data={s:tuple(np.array(a) for a in parts) for s,parts in rows.items()}
    write(out,'dailysports',data,{'task':'classification','source':'UCI 256','source_sha256':sha(raw/'dailysports.zip'),
       'protocol':'p1-p5 train; p6 validation; p7-p8 test; all 19 original classes',
       'notes':'Subject-disjoint; differs from TSCIL processed 18-class variant. Five-second segments never split.'})


def energy(raw,out):
    import csv
    z=zipfile.ZipFile(raw/'energy.zip');name=next(n for n in z.namelist() if n.endswith('.csv'))
    reader=csv.DictReader(io.StringIO(z.read(name).decode()))
    rows=list(reader)
    # Indoor measurements only: omit interpolated weather fields (possible future interpolation),
    # random covariates and contemporaneous/future target values. Past target is permitted.
    cols=['Appliances','lights']+[v for j in range(1,10) for v in ('T'+str(j),'RH_'+str(j))]
    array=np.array([[float(r[c]) for c in cols] for r in rows],dtype='float32')
    n=len(array);data={s:[[],[],[],[]] for s in ('train','val','test')};domains=5
    for domain,chunk in enumerate(np.array_split(np.arange(n),domains)):
        # Split raw timeline first; no input or target overlap across split boundaries.
        splits=np.split(chunk,[int(.6*len(chunk)),int(.8*len(chunk))])
        for split,part in zip(('train','val','test'),splits):
            for offset in range(0,len(part)-24-6+1,30):
                start=int(part[offset]);x=array[start:start+24].T;y=array[start+24:start+30,0].mean()
                data[split][0].append(x);data[split][1].append([y]);data[split][2].append(f'{start}:{start+30}')
                data[split][3].append(domain)
    data={s:tuple(np.array(a) for a in parts) for s,parts in data.items()}
    write(out,'energy',data,{'task':'regression','source':'UCI 374','source_sha256':sha(raw/'energy.zip'),
      'columns':cols,'target':'mean Appliances Wh per 10-min interval over next six intervals, NOT six-interval sum',
      'protocol':'5 chronological domains; each raw timeline split 60/20/20 before 24-step lookback / 6-step future / stride30',
      'notes':'No raw timestamp overlap across windows. Pilot sequence-to-vector regression. Weather interpolation omitted. Not a general forecasting benchmark.'})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--datasets',nargs='+',required=True);args=ap.parse_args()
    for name in args.datasets: {'har':har,'uwave':uwave,'dailysports':sports,'energy':energy}[name](args.root/'raw',args.root/'prepared')

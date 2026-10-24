"""Audit fixed splits and cache provenance without evaluating any TEST prediction."""
import argparse,hashlib,json,zipfile
from pathlib import Path
import numpy as np
from scripts.prepare_data import read_ts,sha
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();rows=[]
for name,archive in (('phonemespectra','PhonemeSpectra'),('lsst','LSST'),('handwriting','Handwriting')):
    meta=json.loads((a.root/'prepared'/f'{name}_manifest.json').read_text())
    assert sha(a.root/'raw/extension'/f'{archive}.zip')==meta['source_sha256']
    with zipfile.ZipFile(a.root/'raw/extension'/f'{archive}.zip') as z:
        x,labels=read_ts(z.read(next(n for n in z.namelist() if n.endswith('_TRAIN.ts'))))
    mapping={c:i for i,c in enumerate(sorted(set(labels)))};assert mapping==meta['classes']
    y=np.array([mapping[c] for c in labels]);rng=np.random.default_rng(1701);valid=[]
    for c in np.unique(y):
        inds=np.flatnonzero(y==c);valid.extend(rng.permutation(inds)[:max(1,len(inds)//5)])
    ids={};hashes={}
    for split in ('train','val','test'):
        source=a.root/'prepared'/f'{name}_{split}.npz';assert sha(source)==meta['files'][split]
        with np.load(source) as d:
            ids[split]=set(d['ids'].tolist());assert len(ids[split])==len(d['x'])
            assert np.isfinite(d['x']).all() and np.isfinite(d['y']).all()
            hashes[split]={hashlib.sha256(row.tobytes()).hexdigest() for row in d['x']}
            if split!='test':
                inds=np.array([int(i.split(':')[1]) for i in d['ids']]);expected=np.isin(np.arange(len(y)),valid)
                if split=='train':expected=~expected
                np.testing.assert_array_equal(inds,np.flatnonzero(expected));np.testing.assert_array_equal(d['x'],x[inds].astype(np.float32));np.testing.assert_array_equal(d['y'],y[inds])
                for encoder in ('mantis','summary'):
                    f=a.root/'features'/encoder/f'{name}_{split}.npz';cache=json.loads(f.with_suffix('.json').read_text());assert cache['source_sha256']==meta['files'][split] and cache['cache_sha256']==sha(f)
                    with np.load(f) as e:
                        assert np.isfinite(e['x']).all() and len(e['x'])==len(d['x'])
                        for k in ('ids','y','groups'):np.testing.assert_array_equal(e[k],d[k])
    for l,r in (('train','val'),('train','test'),('val','test')):assert not ids[l]&ids[r]
    for encoder in ('mantis','summary'):assert not (a.root/'features'/encoder/f'{name}_test.npz').exists()
    rows.append(dict(dataset=name,sizes=meta['sizes'],exact_input_duplicates={l+'_'+r:len(hashes[l]&hashes[r]) for l,r in (('train','val'),('train','test'),('val','test'))},test_predictions_evaluated=False))
report=dict(complete=True,checked=rows);a.out.write_text(json.dumps(report,indent=2));print(json.dumps(report))

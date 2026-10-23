"""Recompute TRAIN/VAL descriptors and verify fusion blocks and sample alignment."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from tmot.temporal_joint import temporal_descriptor
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();checked=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('natops','epilepsy','racketsports','pendigits','articulary','beijing_pm25','air_quality'):
    for split in ('train','val'):
        raw_path=a.root/'prepared'/f'{name}_{split}.npz';old_path=a.root/'features/mantis'/f'{name}_{split}.npz'
        with np.load(raw_path) as raw,np.load(old_path) as old:
            c=raw['x'].shape[1];amp=2*c;desc=np.concatenate([temporal_descriptor(raw['x'][i:i+128]) for i in range(0,len(raw['x']),128)])
            for encoder in ('temporal_joint','mantis_trace','mantis_joint'):
                path=a.root/'features'/encoder/f'{name}_{split}.npz';meta=json.loads(path.with_suffix('.json').read_text())
                assert meta['source_sha256']==sha(raw_path) and meta['mantis_sha256']==sha(old_path) and meta['cache_sha256']==sha(path)
                with np.load(path) as d:
                    for key in ('ids','y','groups'):np.testing.assert_array_equal(d[key],raw[key]);np.testing.assert_array_equal(d[key],old[key])
                    np.testing.assert_array_equal(d['x'][:,-amp:],old['x'][:,-amp:])
                    if encoder=='temporal_joint':np.testing.assert_array_equal(d['x'][:,:-amp],desc)
                    else:
                        start=old['x'].shape[1]-amp;np.testing.assert_array_equal(d['x'][:,:start],old['x'][:,:start])
                        expected=desc[:,:32*c] if encoder=='mantis_trace' else desc
                        np.testing.assert_array_equal(d['x'][:,start:-amp],expected)
                checked.append(str(path))
assert len(checked)==42
a.out.write_text(json.dumps(dict(complete=True,caches=len(checked),checked=checked,scope='TRAIN and VAL only; all descriptors recomputed; original ordered Mantis/amplitude blocks exact; all IDs and labels exact'),indent=2));print('42 TRAIN/VAL feature caches verified')

"""Verify raw temporal input lineage, original feature blocks and IDs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from tmot.raw_temporal import raw_temporal
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();checked=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('natops','epilepsy','racketsports','pendigits','articulary','beijing_pm25','air_quality'):
    for split in ('train','val'):
        raw_path=a.root/'prepared'/f'{name}_{split}.npz';old_path=a.root/'features/mantis'/f'{name}_{split}.npz'
        with np.load(raw_path) as raw,np.load(old_path) as old:
            c=raw['x'].shape[1];amp=2*c;windows=raw_temporal(raw['x']);last=raw_temporal(raw['x'],False)
            for encoder in ('raw_windows','mantis_raw_last','mantis_raw_windows'):
                path=a.root/'features'/encoder/f'{name}_{split}.npz';meta=json.loads(path.with_suffix('.json').read_text())
                assert meta['source_sha256']==sha(raw_path) and meta['mantis_sha256']==sha(old_path) and meta['cache_sha256']==sha(path)
                with np.load(path) as d:
                    for key in ('ids','y','groups'):np.testing.assert_array_equal(d[key],raw[key]);np.testing.assert_array_equal(d[key],old[key])
                    if encoder=='raw_windows':np.testing.assert_array_equal(d['x'],windows)
                    else:
                        start=old['x'].shape[1]-amp;np.testing.assert_array_equal(d['x'][:,:start],old['x'][:,:start]);np.testing.assert_array_equal(d['x'][:,-amp:],old['x'][:,-amp:])
                        np.testing.assert_array_equal(d['x'][:,start:-amp],last if encoder=='mantis_raw_last' else windows)
                checked.append(str(path))
assert len(checked)==42
a.out.write_text(json.dumps(dict(complete=True,caches=len(checked),checked=checked,scope='TRAIN/VAL only, all-channel raw coordinates recomputed and sample alignment exact'),indent=2));print('42 raw temporal feature caches verified')

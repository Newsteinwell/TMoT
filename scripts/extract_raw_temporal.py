"""Extract raw temporal coordinates for TRAIN/VAL only, preserving sample IDs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from tmot.raw_temporal import raw_temporal
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('natops','epilepsy','racketsports','pendigits','articulary','beijing_pm25','air_quality'):
    for split in ('train','val'):
        source=a.root/'prepared'/f'{name}_{split}.npz';old_path=a.root/'features/mantis'/f'{name}_{split}.npz'
        with np.load(source) as d,np.load(old_path) as old:
            for key in ('ids','y','groups'):np.testing.assert_array_equal(d[key],old[key])
            amp=2*d['x'].shape[1];windows=raw_temporal(d['x']);last=raw_temporal(d['x'],False)
            for encoder,desc in [('mantis_raw_last',last),('mantis_raw_windows',windows),('raw_windows',windows)]:
                x=desc if encoder=='raw_windows' else np.concatenate([old['x'][:,:-amp],desc,old['x'][:,-amp:]],axis=1)
                folder=a.root/'features'/encoder;folder.mkdir(parents=True,exist_ok=True);target=folder/f'{name}_{split}.npz'
                if target.exists():raise FileExistsError(target)
                np.savez_compressed(target,x=x,y=d['y'],ids=d['ids'],groups=d['groups'])
                target.with_suffix('.json').write_text(json.dumps(dict(dataset=name,split=split,shape=list(x.shape),raw_dim=desc.shape[1],source_sha256=sha(source),mantis_sha256=sha(old_path),cache_sha256=sha(target),preprocessing='all-channel original-unit last values; optionally eight chronological bin means; no label-based channel selection or cross-sample fit'),indent=2))
                print(name,split,encoder,x.shape,flush=True)

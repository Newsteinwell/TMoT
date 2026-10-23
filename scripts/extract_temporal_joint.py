"""Create only TRAIN/VAL sample-local descriptors; official TEST stays unopened."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from tmot.temporal_joint import temporal_descriptor
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('natops','epilepsy','racketsports','pendigits','articulary','beijing_pm25','air_quality'):
    for split in ('train','val'):
        source=a.root/'prepared'/f'{name}_{split}.npz';old_path=a.root/'features/mantis'/f'{name}_{split}.npz'
        with np.load(source) as d,np.load(old_path) as old:
            for key in ('ids','y','groups'):np.testing.assert_array_equal(d[key],old[key])
            t=time.monotonic();parts=[]
            for start in range(0,len(d['x']),128):parts.append(temporal_descriptor(d['x'][start:start+128]))
            desc=np.concatenate(parts);amp_dim=2*d['x'].shape[1]
            for encoder,features in [('temporal_joint',np.concatenate([desc,old['x'][:,-amp_dim:]],axis=1)),
                                     ('mantis_trace',np.concatenate([old['x'][:,:-amp_dim],desc[:,:32*d['x'].shape[1]],old['x'][:,-amp_dim:]],axis=1)),
                                     ('mantis_joint',np.concatenate([old['x'][:,:-amp_dim],desc,old['x'][:,-amp_dim:]],axis=1))]:
                folder=a.root/'features'/encoder;folder.mkdir(parents=True,exist_ok=True);target=folder/f'{name}_{split}.npz'
                if target.exists():raise FileExistsError(target)
                np.savez_compressed(target,x=features,y=d['y'],ids=d['ids'],groups=d['groups'])
                meta=dict(dataset=name,split=split,encoder=encoder,shape=list(features.shape),seconds=time.monotonic()-t,
                          source_sha256=sha(source),mantis_sha256=sha(old_path),cache_sha256=sha(target),joint_dimension=32*d['x'].shape[1] if encoder=='mantis_trace' else desc.shape[1],preprocessing='sample-local normalized trace32 and cross-channel correlations at lags0,1,2,4; no labels or cross-sample fit')
                target.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(name,split,encoder,features.shape,flush=True)

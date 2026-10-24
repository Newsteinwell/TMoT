"""TRAIN-only structural eligibility check; never evaluate a model or read TEST."""
import argparse,json,zipfile
from pathlib import Path
import numpy as np
from scripts.prepare_data import read_ts,sha
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();rows=[]
for name in ('PhonemeSpectra','LSST','Handwriting'):
    path=a.root/'raw/extension'/f'{name}.zip'
    with zipfile.ZipFile(path) as z:
        member=next(n for n in z.namelist() if n.endswith('_TRAIN.ts'))
        x,y=read_ts(z.read(member));labels,counts=np.unique(y,return_counts=True)
        row=dict(archive=name,sha256=sha(path),shape=list(x.shape),classes=len(labels),class_counts=dict(zip(labels.tolist(),counts.tolist())),finite=bool(np.isfinite(x).all()),test_read=False)
        assert row['finite'] and counts.min()>=2
        rows.append(row);print(json.dumps(row),flush=True)
(a.root/'extension_train_inventory.json').write_text(json.dumps(rows,indent=2))

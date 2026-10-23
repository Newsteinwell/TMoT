"""Causal, no-fit references on round-3 VAL only; never open TEST."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def evaluate(root):
    results = []
    for name in ('beijing_pm25', 'air_quality'):
        folder = root / 'prepared'
        manifest = json.loads((folder / f'{name}_manifest.json').read_text())
        assert manifest['columns'][0] == ('pm2.5' if name == 'beijing_pm25' else 'CO(GT)')
        data = {}
        for split in ('train', 'val'):
            path = folder / f'{name}_{split}.npz'
            assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['files'][split]
            with np.load(path) as archive:
                data[split] = {k: archive[k] for k in archive.files}
        intervals = sorted(tuple(map(int, str(i).split(':'))) for d in data.values() for i in d['ids'])
        assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
        rows = []
        for stage in sorted(np.unique(data['train']['groups'])):
            train = data['train']['groups'] <= stage
            val = data['val']['groups'] <= stage
            x = data['val']['x'][val].astype(np.float64)
            y = data['val']['y'][val].astype(np.float64)
            predictions = {
                'seen_train_mean': np.full_like(y, data['train']['y'][train].mean()),
                'last_value': x[:, 0, -1, None],
                'last_six_mean': x[:, 0, -6:].mean(1)[:, None],
                'last_24_mean': x[:, 0].mean(1)[:, None],
            }
            for method, pred in predictions.items():
                assert pred.shape == y.shape
                rows.append(dict(stage=int(stage)+1, method=method, n=len(y),
                                 rmse=float(np.sqrt(np.mean((pred-y)**2))),
                                 mae=float(np.mean(np.abs(pred-y)))))
        results.append(dict(dataset=name, split='val', hashes={s:manifest['files'][s] for s in data},
                            train_val_nonoverlap=True, stages=rows,
                            average_stage_rmse={m:float(np.mean([r['rmse'] for r in rows if r['method']==m])) for m in predictions}))
    return dict(scope='VAL only; input channel zero in original units; causal filled inputs', datasets=results)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();result=evaluate(args.root)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2))
    for d in result['datasets']:print(d['dataset'], json.dumps(d['stages'][-4:]))

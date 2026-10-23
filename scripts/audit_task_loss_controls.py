"""Confirm 54 unmodified MSE trajectories reproduce previously executed runs."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args();checked=[]
for path in sorted((a.root/'runs/task_loss_v26').glob('*_mse_*/summary.json')):
    new=json.loads(path.read_text());old_path=a.root/'runs/nonlinear_validation_v12'/path.parent.name.replace('_mse_','_');old=json.loads((old_path/'summary.json').read_text())
    assert new['input_hashes']==old['input_hashes'] and new['order']==old['order']
    for stage,ref in zip(new['stages'],old['stages'],strict=True):
        for key in ('optimizer_steps','historical_training_samples','current_samples'):assert stage[key]==ref[key]
        filename=f"predictions_stage{stage['stage']}.npz"
        with np.load(path.parent/filename) as x,np.load(old_path/filename) as y:
            assert set(x.files)==set(y.files)
            for key in x.files:np.testing.assert_array_equal(x[key],y[key])
    checked.append(path.parent.name)
assert len(checked)==54
a.out.write_text(json.dumps(dict(complete=True,independent_controls=len(checked),checked=checked),indent=2));print('54 MSE trajectories exactly reproduced')

"""Reproduce all 42 existing 512-dimensional control trajectories exactly."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
checked=[]
for path in sorted((a.root/'runs/temporal_joint_v27').glob('*_mantis_d512_*/summary.json')):
    run=json.loads(path.read_text());c=run['config']
    old=a.root/'runs/round3_validation_v9'/f"{c['dataset']}_mantis_linear_r{c['ridge']:g}_o17_s42"
    ref=json.loads((old/'summary.json').read_text());assert run['input_hashes']==ref['input_hashes'] and run['order']==ref['order']
    for stage in run['stages']:
        file=f"predictions_stage{stage['stage']}.npz"
        with np.load(path.parent/file) as new,np.load(old/file) as baseline:
            assert set(new.files)==set(baseline.files)
            for key in new.files:np.testing.assert_array_equal(new[key],baseline[key])
    checked.append(path.parent.name)
assert len(checked)==42
a.out.write_text(json.dumps(dict(complete=True,exact_control_trajectories=len(checked),checked=checked),indent=2));print('42 independent control trajectories exactly reproduced')

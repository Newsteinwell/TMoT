"""No-training references in original target units, with temporal split audit."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
data={s:np.load(a.root/'prepared'/f'energy_{s}.npz') for s in ('train','val','test')}
intervals=[]
for split,d in data.items():
    for id_ in d['ids']:
        start,end=map(int,str(id_).split(':'));intervals.append((start,end,split))
intervals.sort()
assert all(intervals[i][1]<=intervals[i+1][0] for i in range(len(intervals)-1)), 'Temporal overlap'
rows=[]
for stage in sorted(np.unique(data['train']['groups'])):
    tr=data['train']['groups']<=stage;te=data['test']['groups']<=stage
    y=data['test']['y'][te]
    preds={'seen_train_mean':np.full_like(y,data['train']['y'][tr].mean()),
           'last_value':data['test']['x'][te,0,-1,None],
           'last_six_mean':data['test']['x'][te,0,-6:].mean(1)[:,None]}
    for name,pred in preds.items():
        rows.append(dict(stage=int(stage)+1,method=name,rmse=float(np.sqrt(np.mean((y-pred)**2))),mae=float(np.mean(np.abs(y-pred)))))
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps({'nonoverlap_audit':'passed','stages':rows},indent=2));print(json.dumps(rows[-3:],indent=2))

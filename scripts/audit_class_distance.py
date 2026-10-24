import json,hashlib,sys
from pathlib import Path
import numpy as np
import torch
root=Path(sys.argv[1]);folder=root/'runs/class_distance_v61';terminal=json.loads((folder/'complete.json').read_text());assert len(terminal)==6 and all(r['status']=='completed' for r in terminal)
checked=[];rows=[];controls=0;max_centering_error=0.
for path in sorted(folder.glob('*/report.json')):
 s=json.loads(path.read_text());assert s['complete'] and s['rng_unchanged'] and s['old_prediction_control'] and s['network_and_points_unchanged'];assert len(s['rows'])==36
 shared=Path(s['source_checkpoint']);assert hashlib.sha256(shared.read_bytes()).hexdigest()==s['source_checkpoint_sha256'];m=torch.load(shared,map_location='cpu',weights_only=False)['learner'];classes=np.array(m.classes)
 for st in s['statistics']:max_centering_error=max(max_centering_error,st['covariance_centering_error'])
 for row in s['rows']:
  d,e=row['dataset'],row['encoder'];val=np.load(root/'features'/e/f'{d}_val.npz')
  for split,digest in s['input_hashes'].items():assert hashlib.sha256((root/'features'/e/f'{d}_{split}.npz').read_bytes()).hexdigest()==digest
  with np.load(path.parent/row['artifact']) as pred:
   assert np.isfinite(pred['distance']).all();np.testing.assert_array_equal(pred['pred'],classes[pred['distance'].argmin(1)])
   assert set(pred['ids'].tolist())==set(val['ids'].tolist()) and len(np.unique(pred['ids']))==len(val['ids'])
   mapping={v:i for i,v in enumerate(val['ids'])};np.testing.assert_array_equal(pred['y'],val['y'][[mapping[v] for v in pred['ids']]])
   metric=float(np.mean([np.mean(pred['pred'][pred['y']==c]==c) for c in np.unique(pred['y'])]));np.testing.assert_allclose(metric,row['final_macro_accuracy'],rtol=1e-12,atol=1e-12)
   if row['fecam_control_exact']:
    old=root/'runs/fecam_harder_v59'/f"{d}_{e}_fecam_a{row['alpha']:g}_o17_s42";os=json.loads((old/'summary.json').read_text())
    with np.load(old/f"predictions_stage{len(os['order'])}.npz") as op:
     np.testing.assert_array_equal(pred['pred'],np.concatenate([op[f'pred_{i}'] for i in range(len(os['order']))]))
    controls+=1
  checked.append(str(path.parent/row['artifact']));rows.append(row)
assert len(rows)==216 and controls==36
(root/'class_distance_v61_audit.json').write_text(json.dumps(dict(complete=True,checked=checked,fecam_controls=controls,source_checkpoints=6,max_centering_error=max_centering_error,checks=['prediction metrics','saved distance argmin','validation IDs and labels','input/checkpoint hashes','FeCAM controls','reported frozen state and RNG checks']),indent=2))
(root/'class_distance_v61_rows.json').write_text(json.dumps(rows,indent=2))
selected=[]
for d,e in sorted({(r['dataset'],r['encoder']) for r in rows}):
 for space in ('input','hidden','joint'):
  for origin in ('synthetic','real_diagnostic'):
   r=min((r for r in rows if (r['dataset'],r['encoder'],r['space'],r['origin'])==(d,e,space,origin)),key=lambda r:(-r['final_macro_accuracy'],r['alpha']));selected.append(r);print(d,e,space,origin,round(100*r['final_macro_accuracy'],3),r['alpha'],flush=True)
(root/'class_distance_v61_selected.json').write_text(json.dumps(selected,indent=2))

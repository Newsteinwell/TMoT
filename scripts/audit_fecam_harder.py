import json,sys
from pathlib import Path
import numpy as np
import torch
root=Path(sys.argv[1]);folder=root/'runs/fecam_harder_v59';torch.set_num_threads(1);rows=[]
for p in sorted(folder.glob('*/summary.json')):
 s=json.loads(p.read_text());c=s['config'];ck=torch.load(p.parent/'final.pt',map_location='cpu',weights_only=False);m=ck['learner']
 val=np.load(root/'features'/c['encoder']/(c['dataset']+'_val.npz'))
 pred=np.load(p.parent/f"predictions_stage{len(s['order'])}.npz")
 for k,group in enumerate(s['order']):
  ids=np.flatnonzero(np.isin(val['y'],group));actual=m.predict(ck['basis'].transform(val['x'][ids]))
  np.testing.assert_array_equal(actual,pred[f'pred_{k}']);np.testing.assert_array_equal(val['y'][ids],pred[f'y_{k}']);np.testing.assert_array_equal(val['ids'][ids],pred[f'ids_{k}'])
 peer=next((root/'runs/task_budget_v57').glob(c['dataset']+'_'+c['encoder']+'_er10_o17_s42_r*/summary.json'))
 old=json.loads(peer.read_text());assert old['order']==s['order'] and old['input_hashes']==s['input_hashes']
 rows.append(s)
assert len(rows)==36
(root/'fecam_harder_v59_runs.json').write_text(json.dumps(rows,indent=2))
(root/'fecam_harder_v59_state_audit.json').write_text(json.dumps(dict(complete=True,checkpoints=36,final_predictions_exact=True,input_and_order_matched=True),indent=2))

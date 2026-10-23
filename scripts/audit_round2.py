"""Recompute saved stage metrics and verify the paired readout experiment."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
report={'queues':{},'paired_readout_state':[]}
for name,expected in [('diagnosis_v2',112),('transport_v3',52),('followups_v4',12),('locked_test_v5',34)]:
    queue=a.root/'runs'/name;checked=[]
    for f in sorted(queue.glob('*/summary.json')):
        s=json.loads(f.read_text());assert (f.parent/'final.pt').exists()
        for stage in s['stages']:
            with np.load(f.parent/f"predictions_stage{stage['stage']}.npz") as d:
                keys=sorted(int(k.split('_')[1]) for k in d.files if k.startswith('y_'))
                ys=np.concatenate([d[f'y_{k}'] for k in keys]);ps=np.concatenate([d[f'pred_{k}'] for k in keys])
                metric=(np.mean([np.mean(ps[ys==c]==c) for c in np.unique(ys)]) if s['task']=='classification'
                        else np.sqrt(np.mean((ys-ps)**2)))
                np.testing.assert_allclose(metric,stage['macro_accuracy' if s['task']=='classification' else 'rmse'],rtol=1e-6,atol=1e-12)
        assert s['config']['eval_split']==('test' if name=='locked_test_v5' else 'val')
        checked.append(f.parent.name)
    report['queues'][name]=dict(expected=expected,completed_metric_checked=len(checked),checked=checked)
queue=a.root/'runs'/'diagnosis_v2'
for f in sorted(queue.glob('*_tmot_d256_*/final.pt')):
    peer=f.parent.with_name(f.parent.name.replace('_tmot_d256_','_tmot_trained_'))/'final.pt'
    if not peer.exists():continue
    left=torch.load(f,map_location='cpu',weights_only=False);right=torch.load(peer,map_location='cpu',weights_only=False)
    for component in ('adapter','head'):
        for key,tensor in getattr(left['learner'],component).state_dict().items():
            assert torch.equal(tensor,getattr(right['learner'],component).state_dict()[key]),(str(f),component,key)
    for attr in ('mean','sd','projection'):
        np.testing.assert_array_equal(getattr(left['basis'],attr),getattr(right['basis'],attr))
    report['paired_readout_state'].append(dict(left=f.parent.name,right=peer.parent.name,exact_weights_and_basis_equal=True))
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2))
print(json.dumps({k:{i:v for i,v in d.items() if i!='checked'} for k,d in report['queues'].items()}))
print('identical readout pairs',len(report['paired_readout_state']))

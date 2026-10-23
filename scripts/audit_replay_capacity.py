"""Inspect actual retention, not nominal buffer size, in existing ER controls."""
import argparse,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
rows=[]
for dataset in ('har','uwave','dailysports','energy','natops','epilepsy','racketsports','pendigits','articulary','beijing_pm25','air_quality'):
    old=dataset in ('har','uwave','dailysports','energy')
    queue='nonlinear_matched_v11' if old else 'nonlinear_validation_v12'
    name=f'{dataset}_er_w1_o17_s42' if old else f'{dataset}_mantis_er_w1_o17_s42'
    folder=a.root/'runs'/queue/name
    model=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)['learner']
    summary=json.loads((folder/'summary.json').read_text());retained=len(model.buffer[0])
    assert sum(s['current_samples'] for s in summary['stages'])==model.seen
    rows.append(dict(dataset=dataset,encoder='mantis',source_run=name,queue=queue,seen=int(model.seen),retained=retained,
                     retained_fraction=retained/model.seen,all_seen_rows_retained=retained==model.seen,
                     buffer_payload_bytes=sum(v.nbytes for v in model.buffer)+model.keys.nbytes,
                     total_learner_persistent_bytes=summary['stages'][-1]['persistent_bytes']))
a.out.write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))

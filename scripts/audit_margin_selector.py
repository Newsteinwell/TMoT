import json,sys,hashlib
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);folder=root/'runs/margin_selector_v66';terminal=json.loads((folder/'complete.json').read_text());assert len(terminal)==36 and all(r['status']=='completed' for r in terminal)
rows=[]
for item in terminal:
 group=item['group'];r=json.loads((folder/(group+'.json')).read_text());c=r['config'];original=root/'runs/bounded_robustness_v64';s=json.loads((original/(group+'_bounded')/'summary.json').read_text())
 with np.load(folder/(group+'.npz')) as data:
  dr=data['reference_distances'];dc=data['candidate_distances'];classes=data['classes'];scores=[]
  for distance in (dr,dc):
   ordered=np.sort(distance,axis=1);den=ordered[:,0]+ordered[:,1];scores.append(np.divide(ordered[:,1]-ordered[:,0],den,out=np.zeros(len(den)),where=den>0))
  choose=scores[1]>scores[0];pred=classes[np.where(choose,dc.argmin(1),dr.argmin(1))];np.testing.assert_array_equal(choose,data['choose_candidate']);np.testing.assert_array_equal(pred,data['prediction'])
  y=data['y'];rp=classes[dr.argmin(1)];cp=classes[dc.argmin(1)]
  for mode,p in [('fecam',rp),('bounded',cp)]:
   with np.load(original/(group+'_'+mode)/f"predictions_stage{len(s['order'])}.npz") as saved:
    for key in ('y','ids'):np.testing.assert_array_equal(data[key],np.concatenate([saved[f'{key}_{j}'] for j in range(len(s['order']))]))
    np.testing.assert_array_equal(p,np.concatenate([saved[f'pred_{j}'] for j in range(len(s['order']))]))
  for key,p in [('reference_accuracy',rp),('candidate_accuracy',pred)]:
   score=np.mean([np.mean(p[y==label]==label) for label in np.unique(y)]);np.testing.assert_allclose(score,r['metrics'][key],atol=1e-12)
  assert abs(r['metrics']['candidate_accuracy']-r['metrics']['reference_accuracy']-r['metrics']['rescue']+r['metrics']['harm'])<1e-12
  r['archive_sha256']=hashlib.sha256((folder/(group+'.npz')).read_bytes()).hexdigest();rows.append(r)
summary=[]
for d,e in sorted({(r['config']['dataset'],r['config']['encoder']) for r in rows}):
 selected=[r for r in rows if (r['config']['dataset'],r['config']['encoder'])==(d,e) and (r['config']['seed'],r['config']['order'])!=(42,17)]
 gains=[100*(r['metrics']['candidate_accuracy']-r['metrics']['reference_accuracy']) for r in selected]
 row=dict(dataset=d,encoder=e,n=len(selected),mean_gain_pp=float(np.mean(gains)),min_gain_pp=min(gains),max_gain_pp=max(gains),wins=sum(g>1e-10 for g in gains),at_least_1pp=sum(g>=1 for g in gains),mean_rescue_pp=float(np.mean([100*r['metrics']['rescue'] for r in selected])),mean_harm_pp=float(np.mean([100*r['metrics']['harm'] for r in selected])),mean_ungated_gain_pp=float(np.mean([100*(r['ungated']['candidate_accuracy']-r['ungated']['reference_accuracy']) for r in selected])))
 summary.append(row);print(json.dumps(row),flush=True)
(root/'margin_selector_v66_comparison.json').write_text(json.dumps(dict(complete=True,final_states=36,rows=rows,summary=summary,checks=['independent sorted-margin rule','exact v64 predictions and identities','macro metrics','source archives'],note='Final-state development diagnostic only; no full-stream or held-out claim.'),indent=2))

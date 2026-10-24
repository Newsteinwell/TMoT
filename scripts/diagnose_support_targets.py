"""Final-stage support provenance and teacher confidence; no model updates."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from tmot.distillation_support import DistillationSupport
from tmot.gradient_diagnostic import empirical_sample
from tmot.fecam import FeCAMLearner
from tmot.learner import MomentLearner
from tmot.statistical_teacher import distance_probabilities

def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def confidence(p):
 entropy=-(p*np.log(np.maximum(p,1e-300))).sum(1)
 return dict(mean_max_probability=float(p.max(1).mean()),mean_entropy=float(entropy.mean()),normalized_entropy=float(entropy.mean()/np.log(p.shape[1])) if p.shape[1]>1 else 0.)
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--dataset',required=True);p.add_argument('--encoder',required=True);p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(1)
root=Path(a.root);d,e=a.dataset,a.encoder;tr=np.load(root/'features'/e/f'{d}_train.npz');va=np.load(root/'features'/e/f'{d}_val.npz');rows=[]
for mode in ('current','local','er10'):
 folder=root/f'runs/support_teacher_v75/{d}_{e}_{mode}';source=folder/'final.pt';before=digest(source);s=json.loads((folder/'summary.json').read_text());state=torch.load(source,map_location='cpu',weights_only=False);cl=s['task']=='classification';basis=state['basis'];mu=state['target_mean'];sd=state['target_sd'];pool=DistillationSupport(512,s['task'],mode);teacher=FeCAMLearner(512,alpha1=10.) if cl else MomentLearner(512,'regression','ridge',42,ridge=.1)
 for t,part in enumerate(s['order']):
  ids=np.flatnonzero(np.isin(tr['y'],part)) if cl else np.flatnonzero(tr['groups']==part);x=basis.transform(tr['x'][ids]);y=tr['y'][ids] if cl else (tr['y'][ids]-mu)/sd;teacher.observe(x,y)
  if t==len(s['order'])-1:
   hx,info=pool.sample(x,y);assert info==s['stages'][-1]['support'];rng=np.random.default_rng(1701+pool.stage)
   if info['historical_support'] and mode=='local':xx,nominal=pool.local.sample(4096,pool.classes,rng)
   elif info['historical_support'] and mode=='er10':
    bx,by=pool.real.buffer;xx,nominal=empirical_sample(bx,by,4096,rng,np.unique(by).tolist() if cl else [])
   else:xx,nominal=empirical_sample(x,y,4096,rng,np.unique(y).tolist() if cl else [])
   np.testing.assert_array_equal(xx,hx)
   actual=state['model'].predict_head(hx);tp=teacher.predict(hx)
   row=dict(dataset=d,encoder=e,task=s['task'],support=mode,source_sha256=before,support_sha256=info['support_sha256'],historical_support=info['historical_support'])
   if cl:
    probs=distance_probabilities(teacher.distances(hx));row.update(confidence(probs));row.update(teacher_nominal_agreement=float(np.mean(tp==nominal)),student_teacher_agreement=float(np.mean(actual==tp)),nominal_classes=len(np.unique(nominal)),teacher_predicted_classes=len(np.unique(tp)),seen_classes=len(teacher.classes),nominal_mean_probability=float(np.mean([probs[i,teacher.classes.index(v)] for i,v in enumerate(nominal)])))
    vinds=np.flatnonzero(~np.isin(va['y'],s['order'][-1]));vp=distance_probabilities(teacher.distances(basis.transform(va['x'][vinds])));row['old_val_teacher_confidence']=confidence(vp)
   else:
    row.update(teacher_nominal_rmse=float(np.sqrt(np.mean(((tp-nominal)*sd)**2))),student_teacher_rmse=float(np.sqrt(np.mean(((actual-tp)*sd)**2))))
   rows.append(row)
  pool.update(x,y)
 assert digest(source)==before
Path(a.out).write_text(json.dumps(dict(source=os.environ['TMOT_SOURCE_COMMIT'],rows=rows),indent=2));print(d,e,'diagnosed',flush=True)

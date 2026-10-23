"""Training-data-only fixed-anchor empirical versus replay gradient diagnosis."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from tmot.gradient_diagnostic import risk_gradient,compare_gradients,empirical_sample,CachedGaussian
from tmot.local_moments import LocalMomentMixture
from tmot.moments import Cell
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('checkpoint_dir',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
summary=json.loads((a.checkpoint_dir/'summary.json').read_text());cfg=summary['config']
checkpoint=a.checkpoint_dir/'final.pt';digest=sha(checkpoint);ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
m=ck['learner'];m.device='cuda';m.net=m.net.to('cuda').eval()
path=a.root/'features'/cfg['encoder']/f"{cfg['dataset']}_train.npz"
with np.load(path) as d:
    x=ck['basis'].transform(d['x']).astype(np.float32);y=d['y'].copy()
    groups=[np.flatnonzero(np.isin(y,c)) for c in summary['order']] if m.task=='classification' else [np.flatnonzero(d['groups']==c) for c in summary['order']]
if m.task=='regression':y=(y-ck['target_mean'])/ck['target_sd']
local=LocalMomentMixture(m.dimension,m.task,seed=718,components=4,preserve_targets=True,support=4,min_cell=4)
for ids in groups:local.update(x[ids],y[ids])
if m.task=='classification':
    gaussian=[CachedGaussian(m.memory.cells[c]) for c in m.classes];weights=np.ones(len(m.classes))/len(m.classes)
else:
    gaussian=[CachedGaussian(Cell.from_x(np.c_[x[ids],y[ids]])) for ids in groups];weights=np.array([len(ids) for ids in groups])/len(x)
# Use the retained ER buffer from the paired, already audited v31 checkpoint.
er_dir=a.checkpoint_dir.parent/a.checkpoint_dir.name.replace('_gaussian_','_er_')
er_checkpoint=er_dir/'final.pt';er=torch.load(er_checkpoint,map_location='cpu',weights_only=False)['learner'];buffer=er.buffer
assert buffer is not None
assert er.classes==m.classes
np.testing.assert_array_equal(er.base,m.base)
assert sha(path)==summary['input_hashes']['train']
before={k:v.detach().cpu().clone() for k,v in m.net.state_dict().items()}
real_loss,reference=risk_gradient(m,x,y)
check_loss,check_grad=risk_gradient(m,x,y,batch=511)
np.testing.assert_allclose(check_grad,reference,rtol=3e-4,atol=2e-6)
rows=[]
for count in (4096,16384):
    for repeat in range(5):
        for method in ('empirical_mc','er','gaussian','local_merged'):
            rng=np.random.default_rng(8900+repeat+count)
            if method=='empirical_mc':sx,sy=empirical_sample(x,y,count,rng,m.classes)
            elif method=='er':sx,sy=empirical_sample(*buffer,count,rng,m.classes)
            elif method=='local_merged':sx,sy=local.sample(count,m.classes,rng)
            else:
                sizes=rng.multinomial(count,weights);xs=[];ys=[]
                for i,(cell,n) in enumerate(zip(gaussian,sizes)):
                    if not n:continue
                    z=cell.sample(n,rng);xs.append(z[:,:m.dimension]);ys.append(np.full(n,m.classes[i]) if m.task=='classification' else z[:,m.dimension:])
                sx,sy=np.concatenate(xs),np.concatenate(ys)
            loss,g=risk_gradient(m,sx,sy)
            rows.append(dict(method=method,count=count,repeat=repeat,loss=loss,loss_difference=loss-real_loss,**compare_gradients(reference,g)))
for k,v in m.net.state_dict().items():torch.testing.assert_close(v.cpu(),before[k],rtol=0,atol=0)
assert sha(checkpoint)==digest
a.out.write_text(json.dumps(dict(complete=True,scope='all arrived training records at final frozen checkpoint; offline diagnosis, no VAL/TEST read or parameter updates',dataset=cfg['dataset'],encoder=cfg['encoder'],loss=m.loss,anchor=m.mode,source=os.environ['TMOT_SOURCE_COMMIT'],checkpoint_sha256=digest,er_checkpoint_sha256=sha(er_checkpoint),train_sha256=sha(path),train_rows=len(x),real_loss=real_loss,reference_norm=float(np.linalg.norm(reference)),gradient_chunk_error=float(np.max(np.abs(reference-check_grad))),rows=rows),indent=2))
print(f'{a.out.name}: 40 gradient comparisons complete',flush=True)

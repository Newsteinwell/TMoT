"""Read-only full-trajectory decomposition and pre-fixed regression mixtures."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from scripts.diagnose_readout_complementarity import decomposition

def residual_decomposition(y,reference,candidate):
 y=np.asarray(y,np.float64);r=np.asarray(reference,np.float64);c=np.asarray(candidate,np.float64)
 if y.shape!=r.shape or y.shape!=c.shape or not y.size:raise ValueError('Aligned nonempty outputs required')
 a=r-y;b=c-y;delta=c-r;aa=float(np.mean(a*a));bb=float(np.mean(b*b));cross=float(np.mean(a*b));den=float(np.mean(delta*delta));beta=float(np.clip(-np.mean(a*delta)/den,0,1)) if den>0 else 0.
 values={str(w):float(np.sqrt(np.mean((a+w*delta)**2))) for w in (.25,.5,.75)}
 return dict(reference_rmse=aa**.5,candidate_rmse=bb**.5,residual_cosine=cross/(aa*bb)**.5 if aa*bb>0 else None,fixed_mixtures=values,label_oracle_beta=beta,label_oracle_rmse=float(np.sqrt(np.mean((a+beta*delta)**2))))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run(a):
 root=Path(a.root);manifest=Path(a.manifest);jobs=json.loads(manifest.read_text());out=Path(a.out);out.mkdir(exist_ok=False);sources={};rows=[]
 for job in jobs:
  candidate=root/job['candidate'];summary=json.loads((candidate/'summary.json').read_text());stages=[]
  for t in range(1,len(summary['stages'])+1):
   cp=candidate/f'predictions_stage{t}.npz';sources[str(cp)]=sha(cp);p=np.load(cp);refdir=job.get('reference_folder');rp=root/refdir/f'predictions_stage{t}.npz' if refdir else cp
   sources[str(rp)]=sha(rp);q=np.load(rp);ys=[];cs=[];rs=[]
   for k in range(t):
    for key in ('ids','y'):np.testing.assert_array_equal(p[f'{key}_{k}'],q[f'{key}_{k}'])
    ys.append(p[f'y_{k}']);cs.append(p[f'pred_{k}']);rs.append(q[f'pred_{k}'] if refdir else q[f'teacher_{k}'])
   y,c,r=map(np.concatenate,(ys,cs,rs));row=decomposition(y,r,c) if summary['task']=='classification' else residual_decomposition(y,r,c)
   metric='macro_accuracy' if summary['task']=='classification' else 'rmse';actual=row['candidate_accuracy' if metric=='macro_accuracy' else 'candidate_rmse'];np.testing.assert_allclose(actual,summary['stages'][t-1][metric],rtol=0,atol=1e-12)
   row['stage']=t;stages.append(row)
  rows.append(dict(**job,task=summary['task'],stages=stages));print(job['dataset'],job['encoder'],job['mode'],job['reference'],'done',flush=True)
 assert all(sha(Path(p))==value for p,value in sources.items())
 (out/'result.json').write_text(json.dumps(dict(source=os.environ['TMOT_SOURCE_COMMIT'],manifest_sha256=sha(manifest),source_hashes=sources,rows=rows,source_files_unchanged=True),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);run(p.parse_args())

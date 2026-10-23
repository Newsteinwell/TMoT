"""Predeclared validation-only dimension/readout diagnosis; no test access."""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
root=a.root;out=root/'runs'/'diagnosis_v2';out.mkdir(parents=True,exist_ok=True)
attempt=f'workers{a.workers}_{os.getpid()}'
variants=[dict(name=f'tmot_d{d}',bottleneck=d) for d in (64,128,256)]
variants += [dict(name='tmot_trained',bottleneck=256,readout='trained'),
             dict(name='tmot_mse',bottleneck=256,current_loss='mse'),
             dict(name='tmot_no_sc',bottleneck=256,sc_weight=0),
             dict(name='ridge',method='ridge',bottleneck=256),
             dict(name='er',method='er',bottleneck=256)]
jobs=[(d,v,o,s) for d in ('har','uwave','dailysports','energy') for v in variants
      for o in ((17,) if d=='energy' else (17,29)) for s in (42,43)]
(out/(attempt+'_manifest.json')).write_text(json.dumps(dict(jobs=jobs,evaluation='val',pid=os.getpid(),workers=a.workers),indent=2))
def run(job):
    dataset,variant,order,seed=job;name=f'{dataset}_{variant["name"]}_o{order}_s{seed}';dest=out/name
    if dest.exists():return dict(name=name,status='existing_not_retried')
    cmd=[sys.executable,'-m','tmot.experiment','--root',str(root),'--out',str(dest),'--dataset',dataset,
         '--eval-split','val','--order',str(order),'--seed',str(seed)]
    for k,v in variant.items():
        if k!='name':cmd.extend(['--'+k.replace('_','-'),str(v)])
    start=time.time()
    with (out/(name+'.log')).open('w') as f:
        try:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=14400);status='completed' if r.returncode==0 else 'failed'
        except subprocess.TimeoutExpired:status='timeout'
    return dict(name=name,status=status,seconds=time.time()-start,command=cmd)
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
    for result in pool.map(run,jobs):
        results.append(result);print(json.dumps(result),flush=True)
        with (out/(attempt+'_status.jsonl')).open('a') as f:f.write(json.dumps(result)+'\n')
(out/(attempt+'_complete.json')).write_text(json.dumps(results,indent=2))

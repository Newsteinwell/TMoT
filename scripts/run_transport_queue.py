"""Parallel validation of migrated modules. Each job has an exclusive directory."""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
out=a.root/'runs'/'transport_v3';out.mkdir(parents=True,exist_ok=True)
variants={'full':[],'ridge':['--readout','ridge'],'no_hist':['--hist-weight','0'],
          'no_pca':['--no-pca'],'no_amplitude':['--no-amplitude'],'frozen':['--frozen'],'no_new_ce':['--no-new-ce']}
jobs=[(d,v,s) for d in ('har','uwave','dailysports','energy') for v in variants
      if not (d=='energy' and v in ('ridge','no_new_ce')) for s in (42,43)]
(out/'manifest.json').write_text(json.dumps(dict(jobs=jobs,pid=os.getpid(),workers=a.workers,variants=variants,selection='val_only',source=os.environ['TMOT_SOURCE_COMMIT']),indent=2))
def run(job):
    d,v,s=job;name=f'{d}_{v}_o17_s{s}';dest=out/name
    if dest.exists():return dict(name=name,status='existing_not_retried')
    cmd=[sys.executable,'-m','tmot.experiment_transport','--root',str(a.root),'--out',str(dest),
         '--dataset',d,'--eval-split','val','--seed',str(s),'--order','17']+variants[v]
    start=time.time()
    with (out/(name+'.log')).open('w') as f:
        try:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=14400);status='completed' if r.returncode==0 else 'failed'
        except subprocess.TimeoutExpired:status='timeout'
    return dict(name=name,status=status,seconds=time.time()-start,command=cmd)
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
    futures=[pool.submit(run,j) for j in jobs]
    for f in concurrent.futures.as_completed(futures):
        result=f.result();results.append(result);print(json.dumps(result),flush=True)
        with (out/'status.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
(out/'complete.json').write_text(json.dumps(results,indent=2))

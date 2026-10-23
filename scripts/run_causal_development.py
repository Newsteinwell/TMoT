"""Development-validation queue; no new benchmark TEST scores are read."""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--workers',type=int,default=16);p.add_argument('--wait-pid',type=int,default=0);a=p.parse_args()
if a.wait_pid:
    path=Path(f'/proc/{a.wait_pid}/cmdline')
    while path.exists() and b'extract_features.py' in path.read_bytes():time.sleep(5)
out=a.root/'runs'/'causal_development_v19';out.mkdir(parents=True,exist_ok=True)
jobs=[]
for dataset in ('beijing_pm25','air_quality'):
    for encoder in ('summary','mantis'):
        for reference in ('none','append','residual'):
            for ridge in (.001,.01,.1,1.,10.,100.):
                for method in ('ridge','stable_residual'):
                    jobs.append((dataset,f'{encoder}_{reference}_{method}_r{ridge:g}',method,ridge,42,
                                 ['--encoder',encoder,'--reference',reference,'--threads','1']))
(out/'manifest.json').write_text(json.dumps(dict(jobs=jobs,pid=os.getpid(),workers=a.workers,source=os.environ['TMOT_SOURCE_COMMIT'],selection='existing development VAL only'),indent=2))
def run(job):
    d,v,m,ridge,seed,args=job;name=f'{d}_{v}_o17_s{seed}';dest=out/name
    if dest.exists():return dict(name=name,status='existing_not_retried')
    cmd=[sys.executable,'-m','tmot.experiment_causal','--root',str(a.root),'--out',str(dest),
         '--dataset',d,'--method',m,'--seed',str(seed),'--order','17','--eval-split','val','--ridge',str(ridge)]+args
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

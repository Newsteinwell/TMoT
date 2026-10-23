"""Matched six-value regularization budgets, existing development validation only."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);out=root/'runs'/'kernel_development_v8';out.mkdir(parents=True,exist_ok=True)
jobs=[(d,v,r) for d in ('har','uwave','dailysports','energy') for v in ('transport','static') for r in (.001,.01,.1,1.,10.,100.)]
(out/'manifest.json').write_text(json.dumps(dict(jobs=jobs,workers=12,pid=os.getpid(),source=os.environ['TMOT_SOURCE_COMMIT'],seed=42,order=17),indent=2))
def run(job):
    d,v,r=job;name=f'{d}_{v}_r{r:g}_o17_s42';dest=out/name
    if dest.exists():return dict(name=name,status='existing_not_retried')
    cmd=[sys.executable,'-m','tmot.experiment_kernel','--root',str(root),'--out',str(dest),'--dataset',d,'--ridge',str(r),'--eval-split','val','--device','cpu','--threads','2']
    if v=='static':cmd+=['--no-transport']
    start=time.time()
    with (out/(name+'.log')).open('w') as f:
        try:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=14400);status='completed' if r.returncode==0 else 'failed'
        except subprocess.TimeoutExpired:status='timeout'
    return dict(name=name,status=status,seconds=time.time()-start,command=cmd)
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
    for result in pool.map(run,jobs):
        results.append(result);print(json.dumps(result),flush=True)
        with (out/'status.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
(out/'complete.json').write_text(json.dumps(results,indent=2))

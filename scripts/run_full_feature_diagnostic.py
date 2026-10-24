"""24 independent feature fits, sharing decompositions across 144 ridge trajectories."""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
RIDGES=(.001,.01,.1,1.,10.,100.)
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--workers',type=int,default=12);a=p.parse_args()
out=a.root/'runs/full_feature_v46';out.mkdir(parents=True,exist_ok=False);groups=[];jobs=[]
for dataset in ('phonemespectra','lsst','handwriting','energy','beijing_pm25','air_quality'):
    for encoder in ('mantis','summary'):
        for mode in ('projected','full'):
            name=f'{dataset}_{encoder}_{mode}';cmd=[sys.executable,'-m','tmot.experiment_full_feature','--root',str(a.root),'--out',str(out),'--dataset',dataset,'--encoder',encoder,'--mode',mode]
            rows=[dict(name=f'{name}_r{r:g}_o17_s42',command=cmd) for r in RIDGES];groups.append(dict(name=name,command=cmd,jobs=rows));jobs+=rows
(out/'manifest.json').write_text(json.dumps(dict(jobs=jobs,groups=groups,workers=a.workers,pid=os.getpid(),source=os.environ['TMOT_SOURCE_COMMIT']),indent=2))
def run(group):
    start=time.monotonic()
    with (out/(group['name']+'.log')).open('w') as log:
        try:
            r=subprocess.run(group['command'],stdout=log,stderr=subprocess.STDOUT,timeout=14400,env={**os.environ,'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'})
            status='completed' if r.returncode==0 else 'failed'
        except subprocess.TimeoutExpired:status='timeout'
    return [dict(j,status=status,shared_fit_seconds=time.monotonic()-start) for j in group['jobs']]
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
    for future in concurrent.futures.as_completed([ex.submit(run,g) for g in groups]):
        rows=future.result();results+=rows
        with (out/'status.jsonl').open('a') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        print(json.dumps(dict(group=rows[0]['name'],status=rows[0]['status'],completed=len(results))),flush=True)
(out/'complete.json').write_text(json.dumps(results,indent=2))

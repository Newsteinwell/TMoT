"""Fixed 25 shared fits / 150 readout trajectories; VAL only."""
import concurrent.futures,json,os,shutil,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);assert shutil.disk_usage(root).free>12*1024**3
out=root/'runs/latent_replay_v47';out.mkdir(parents=True,exist_ok=False);groups=[];jobs=[]
for dataset in ('phonemespectra','lsst','handwriting','energy','beijing_pm25','air_quality'):
    for encoder in ('mantis','summary'):
        variants=['gaussian','rademacher']+(['er'] if dataset=='energy' and encoder=='summary' else [])
        for variant in variants:
            name=f'{dataset}_{encoder}_{variant}';cmd=[sys.executable,'-m','tmot.experiment_latent_replay','--root',str(root),'--out',str(out),'--dataset',dataset,'--encoder',encoder,'--variant',variant]
            rows=[dict(name=f'{name}_r{r:g}_o17_s42',command=cmd) for r in (.001,.01,.1,1.,10.,100.)];groups.append(dict(name=name,command=cmd,jobs=rows));jobs+=rows
(out/'manifest.json').write_text(json.dumps(dict(jobs=jobs,groups=groups,pid=os.getpid(),workers=12,source=os.environ['TMOT_SOURCE_COMMIT']),indent=2))
def run(group):
    start=time.monotonic()
    with (out/(group['name']+'.log')).open('w') as f:
        try:
            p=subprocess.run(group['command'],stdout=f,stderr=subprocess.STDOUT,timeout=14400,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OMP_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'})
            status='completed' if p.returncode==0 else 'failed'
        except subprocess.TimeoutExpired:status='timeout'
    return [dict(j,status=status,shared_fit_seconds=time.monotonic()-start) for j in group['jobs']]
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
    for future in concurrent.futures.as_completed([ex.submit(run,g) for g in groups]):
        rows=future.result();results+=rows
        with (out/'status.jsonl').open('a') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
        print(json.dumps(dict(group=rows[0]['name'],status=rows[0]['status'],completed=len(results))),flush=True)
(out/'complete.json').write_text(json.dumps(results,indent=2))

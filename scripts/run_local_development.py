"""Predeclared 72-run local-mixture nonlinear mechanism screen on existing development VAL."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);out=root/'runs'/'local_development_v14';out.mkdir(parents=True,exist_ok=False)
jobs=[]
for dataset,encoder,ridge in [('har','mantis',1.),('uwave','mantis',.1),('dailysports','mantis',.1),('energy','mantis',1.),('beijing_pm25','summary',.1),('air_quality','summary',1.)]:
    for seed in (42,43):
        for components in (1,4):
            for weight in (.01,.1,1.):
                name=f'{dataset}_{encoder}_local_k{components}_w{weight:g}_o17_s{seed}'
                cmd=[sys.executable,'-m','tmot.experiment_nonlinear','--root',str(root),'--out',str(out/name),
                     '--dataset',dataset,'--encoder',encoder,'--method','local','--local-components',str(components),
                     '--residual-weight',str(weight),'--ridge',str(ridge),
                     '--seed',str(seed),'--order','17','--epochs','30','--eval-split','val','--threads','1','--device','cuda','--er-resample']
                jobs.append(dict(name=name,command=cmd))
(out/'manifest.json').write_text(json.dumps(dict(jobs=jobs,workers=16,pid=os.getpid(),source=os.environ['TMOT_SOURCE_COMMIT'],selection='six diagnostic dataset/encoder pairs; VAL only'),indent=2))
def run(job):
    start=time.monotonic()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (out/(job['name']+'.log')).open('w') as log:
        try:
            result=subprocess.run(job['command'],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=14400)
            status='completed' if result.returncode==0 else 'failed'
        except subprocess.TimeoutExpired:status='timeout'
    return dict(job,status=status,seconds=time.monotonic()-start)
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    for future in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
        result=future.result();results.append(result);print(json.dumps(result),flush=True)
        with (out/'status.jsonl').open('a') as log:log.write(json.dumps(result)+'\n')
(out/'complete.json').write_text(json.dumps(results,indent=2))

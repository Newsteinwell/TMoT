"""Fixed 56-run VAL projection diagnostic; does not change encoders or splits."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);out=root/'runs/history_scale_v42';out.mkdir(parents=True,exist_ok=False)
jobs=[]
for dataset in ('har','uwave','dailysports','energy'):
    for encoder in ('mantis',):
        for mode,ridge,regularized in [(m,r,True) for m in ('local','er') for r in (.001,.01,.1,1.,10.,100.)]+[(m,.1,False) for m in ('local','er')]:
            name=f'{dataset}_{encoder}_{mode}_reg{int(regularized)}_r{ridge:g}_o17_s42'
            cmd=[sys.executable,'-m','tmot.experiment_regularized_innovation','--root',str(root),'--out',str(out/name),
                 '--dataset',dataset,'--encoder',encoder,'--method',mode,'--dimension','512',
                 '--ridge',str(ridge),'--eval-split','val','--order','17','--seed','42','--device','cuda','--threads','1']
            if regularized:cmd.append('--regularized')
            jobs.append(dict(name=name,command=cmd))
(out/'manifest.json').write_text(json.dumps(dict(jobs=jobs,workers=16,pid=os.getpid(),source=os.environ['TMOT_SOURCE_COMMIT'],selection='original four development VAL; no TEST; formal seven unchanged'),indent=2))
def run(job):
    start=time.monotonic();env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (out/(job['name']+'.log')).open('w') as log:
        try:
            r=subprocess.run(job['command'],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=14400)
            status='completed' if r.returncode==0 else 'failed'
        except subprocess.TimeoutExpired:status='timeout'
    return dict(job,status=status,seconds=time.monotonic()-start)
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    for future in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
        r=future.result();results.append(r);print(json.dumps(r),flush=True)
        with (out/'status.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
(out/'complete.json').write_text(json.dumps(results,indent=2))

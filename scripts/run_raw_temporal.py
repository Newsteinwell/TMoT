"""Fixed 168-run VAL projection diagnostic; does not change encoders or splits."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);out=root/'runs/raw_temporal_v28';out.mkdir(parents=True,exist_ok=False)
jobs=[]
for dataset in ('natops','epilepsy','racketsports','pendigits','articulary','beijing_pm25','air_quality'):
    for encoder in ('mantis','mantis_raw_last','mantis_raw_windows','raw_windows'):
        dimension=512
        for ridge in (.001,.01,.1,1.,10.,100.):
            name=f'{dataset}_{encoder}_d{dimension}_ridge_r{ridge:g}_o17_s42'
            cmd=[sys.executable,'-m','tmot.experiment_raw_temporal','--root',str(root),'--out',str(out/name),
                 '--dataset',dataset,'--encoder',encoder,'--method','ridge','--dimension',str(dimension),
                 '--ridge',str(ridge),'--eval-split','val','--order','17','--seed','42','--device','cpu','--threads','1']
            jobs.append(dict(name=name,command=cmd))
(out/'manifest.json').write_text(json.dumps(dict(jobs=jobs,workers=12,pid=os.getpid(),source=os.environ['TMOT_SOURCE_COMMIT'],selection='VAL representation diagnosis; no TEST; all seven retained'),indent=2))
def run(job):
    start=time.monotonic();env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    with (out/(job['name']+'.log')).open('w') as log:
        try:
            r=subprocess.run(job['command'],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=14400)
            status='completed' if r.returncode==0 else 'failed'
        except subprocess.TimeoutExpired:status='timeout'
    return dict(job,status=status,seconds=time.monotonic()-start)
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
    for future in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
        r=future.result();results.append(r);print(json.dumps(r),flush=True)
        with (out/'status.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
(out/'complete.json').write_text(json.dumps(results,indent=2))

"""144-run direct expert screen across classification and regression."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);out=root/'runs'/'prior_expert_development_v18';out.mkdir(parents=True,exist_ok=False)
jobs=[]
for dataset,encoder in [('har','mantis'),('uwave','mantis'),('dailysports','mantis'),('energy','mantis'),('beijing_pm25','summary'),('air_quality','summary')]:
    for components in (1,4):
        for density in ('gaussian','student'):
            for ridge in (.001,.01,.1,1.,10.,100.):
                name=f'{dataset}_{encoder}_k{components}_{density}_r{ridge:g}_o17_s42'
                cmd=[sys.executable,'-m','tmot.experiment_prior_experts','--root',str(root),'--out',str(out/name),
                     '--dataset',dataset,'--encoder',encoder,'--components',str(components),'--density',density,'--ridge',str(ridge),
                     '--eval-split','val','--device','cpu','--threads','1','--order','17','--seed','42','--global-ridge',str({'har':1.,'uwave':.1,'dailysports':.1,'energy':1.,'beijing_pm25':.1,'air_quality':1.}[dataset])]
                jobs.append(dict(name=name,command=cmd))
(out/'manifest.json').write_text(json.dumps(dict(jobs=jobs,workers=16,pid=os.getpid(),source=os.environ['TMOT_SOURCE_COMMIT'],selection='six diagnostic dataset/encoder pairs; VAL only'),indent=2))
def run(job):
    start=time.monotonic()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
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

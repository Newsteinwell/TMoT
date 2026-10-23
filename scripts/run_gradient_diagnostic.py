import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);out=root/'gradient_v35';out.mkdir(exist_ok=False);jobs=[]
for dataset in ('natops','epilepsy','racketsports','pendigits','articulary','beijing_pm25','air_quality'):
    for encoder in (('mantis','summary') if dataset in ('beijing_pm25','air_quality') else ('mantis',)):
        for loss in ('mse','task'):
            for anchor in ('gaussian','er'):
                name=f'{dataset}_{encoder}_{loss}_{anchor}_warm1_o17_s42'
                jobs.append(dict(name=name,command=[sys.executable,'scripts/diagnose_gradients.py',str(root),str(root/'runs/warm_transport_v31'/name),str(out/(name+'.json'))]))
(out/'manifest.json').write_text(json.dumps(dict(jobs=jobs,workers=8,source=os.environ['TMOT_SOURCE_COMMIT']),indent=2))
def run(job):
    start=time.monotonic();env=dict(os.environ,PYTHONPATH='.',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (out/(job['name']+'.log')).open('w') as log:
        try:r=subprocess.run(job['command'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=14400);status='completed' if r.returncode==0 else 'failed'
        except subprocess.TimeoutExpired:status='timeout'
    return dict(name=job['name'],status=status,seconds=time.monotonic()-start)
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    for future in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
        r=future.result();results.append(r);print(json.dumps(r),flush=True)
        with (out/'status.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
(out/'complete.json').write_text(json.dumps(results,indent=2))

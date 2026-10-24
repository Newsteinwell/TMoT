import concurrent.futures,json,os,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);out=root/'runs/prequential_reference_v70';out.mkdir(exist_ok=False);env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
jobs=[dict(name=f'{d}_{e}',command=[sys.executable,'-m','tmot.experiment_prequential_reference','--root',str(root),'--out',str(out),'--dataset',d,'--encoder',e]) for d in ('beijing_pm25','air_quality') for e in ('mantis','summary')]
(out/'manifest.json').write_text(json.dumps(dict(jobs=jobs,workers=4,pid=os.getpid(),source=os.environ['TMOT_SOURCE_COMMIT']),indent=2))
def run(j):
 with (out/(j['name']+'.log')).open('x') as log:p=subprocess.run(j['command'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=7200)
 return dict(**j,status='completed' if p.returncode==0 else 'failed',returncode=p.returncode)
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for f in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
  r=f.result();results.append(r);print(r['name'],r['status'],flush=True)
  with (out/'status.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
(out/'complete.json').write_text(json.dumps(results,indent=2))

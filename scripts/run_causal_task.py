import concurrent.futures,json,os,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);assert shutil.disk_usage(root).free>1024**3;out=root/'runs/causal_task_v68';out.mkdir(exist_ok=False)
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
jobs=[]
for d in ('beijing_pm25','air_quality'):
 for e in ('mantis','summary'):
  for ref in ('none','append','residual'):
   name=f'{d}_{e}_{ref}';cmd=[sys.executable,'-m','tmot.experiment_causal_task','--root',str(root),'--out',str(out),'--dataset',d,'--encoder',e,'--reference',ref];jobs.append(dict(name=name,command=cmd))
(out/'manifest.json').write_text(json.dumps(dict(jobs=jobs,workers=12,pid=os.getpid(),source=os.environ['TMOT_SOURCE_COMMIT']),indent=2))
def run(j):
 with (out/(j['name']+'.log')).open('x') as log:
  p=subprocess.run(j['command'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=7200)
 return dict(**j,status='completed' if p.returncode==0 else 'failed',returncode=p.returncode)
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
 for f in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
  r=f.result();results.append(r);print(r['name'],r['status'],flush=True)
  with (out/'status.jsonl').open('a') as log:log.write(json.dumps(r)+'\n')
(out/'complete.json').write_text(json.dumps(results,indent=2))

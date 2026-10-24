import concurrent.futures,json,os,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);out=root/'runs/certified_v82';out.mkdir(exist_ok=False)
assert shutil.disk_usage(root).free>8*1024**3
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
jobs=[]
for d in ('handwriting','lsst','phonemespectra','air_quality','beijing_pm25'):
 for e in ('mantis','summary'):
  name=f'{d}_{e}';jobs.append(dict(name=name,command=[sys.executable,'-m','tmot.experiment_certified','--root',str(root),'--out',str(out/name),'--dataset',d,'--encoder',e]))
(out/'manifest.json').write_text(json.dumps(dict(jobs=jobs,workers=6,pid=os.getpid(),source=env['TMOT_SOURCE_COMMIT']),indent=2))
def run(j):
 with (out/(j['name']+'.log')).open('x') as log:
  try:
   p=subprocess.run(j['command'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=21600)
   return dict(**j,status='completed' if p.returncode==0 else 'failed',returncode=p.returncode)
  except subprocess.TimeoutExpired:return dict(**j,status='timeout')
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 for f in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
  result=f.result();results.append(result);print(result['name'],result['status'],flush=True)
  with (out/'status.jsonl').open('a') as log:log.write(json.dumps(result)+'\n')
(out/'complete.json').write_text(json.dumps(results,indent=2))

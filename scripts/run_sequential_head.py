"""Six development datasets, two encoders, ordinary/moment ER10, fixed ridge grid."""
import concurrent.futures,json,os,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);assert shutil.disk_usage(root).free>2*1024**3
folder=root/'runs/sequential_head_v73';folder.mkdir(exist_ok=False)
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
jobs=[]
for d in ('phonemespectra','lsst','handwriting','energy','beijing_pm25','air_quality'):
 for e in ('mantis','summary'):
  for v in (0.,.1,1.,10.):
   name=f'{d}_{e}_kd{v:g}'
   cmd=[sys.executable,'-m','tmot.experiment_sequential_head','--root',str(root),'--out',str(folder/name),'--dataset',d,'--encoder',e,'--strength',str(v)]
   jobs.append(dict(name=name,command=cmd))
(folder/'manifest.json').write_text(json.dumps(dict(jobs=jobs,pid=os.getpid(),workers=8,source=os.environ['TMOT_SOURCE_COMMIT']),indent=2))
def run(j):
 with (folder/(j['name']+'.log')).open('x') as log:
  try:
   p=subprocess.run(j['command'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=14400)
   return dict(**j,status='completed' if p.returncode==0 else 'failed',returncode=p.returncode)
  except subprocess.TimeoutExpired:return dict(**j,status='timeout')
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
 for f in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
  item=f.result();results.append(item)
  with (folder/'status.jsonl').open('a') as out:out.write(json.dumps(item)+'\n')
  print(item['name'],item['status'],flush=True)
(folder/'complete.json').write_text(json.dumps(results,indent=2))

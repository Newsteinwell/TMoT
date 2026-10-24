import concurrent.futures,json,os,shutil,subprocess,sys
from pathlib import Path
initialization=sys.argv[2] if len(sys.argv)>2 else 'warm';assert initialization in ('warm','cold')
root=Path(sys.argv[1]);assert shutil.disk_usage(root).free>5*1024**3;out=root/'runs'/('causal_residual_v79' if initialization=='warm' else 'cold_residual_v80');out.mkdir(exist_ok=False)
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
ridge_map={('air_quality','mantis'):[.1,.1,.1],('air_quality','summary'):[1,1,1],('beijing_pm25','mantis'):[.01,.1,1],('beijing_pm25','summary'):[.1,.1,10]}
jobs=[]
for (d,e),ridges in ridge_map.items():
 for ref,r in zip(('none','append','residual'),ridges):
  for mode in ('local','er10'):
   for penalty in (.1,1.):
    name=f'{d}_{e}_{ref}_{mode}_p{penalty:g}';cmd=[sys.executable,'-m','tmot.experiment_causal_residual','--root',str(root),'--out',str(out/name),'--dataset',d,'--encoder',e,'--reference',ref,'--ridge',str(r),'--mode',mode,'--penalty',str(penalty),'--initialization',initialization];jobs.append(dict(name=name,command=cmd))
(out/'manifest.json').write_text(json.dumps(dict(jobs=jobs,workers=12,pid=os.getpid(),source=os.environ['TMOT_SOURCE_COMMIT']),indent=2))
def run(j):
 with (out/(j['name']+'.log')).open('x') as log:
  try:
   p=subprocess.run(j['command'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=7200)
   return dict(**j,status='completed' if p.returncode==0 else 'failed',returncode=p.returncode)
  except subprocess.TimeoutExpired:return dict(**j,status='timeout')
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
 for f in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
  r=f.result();results.append(r);print(r['name'],r['status'],flush=True)
  with (out/'status.jsonl').open('a') as log:log.write(json.dumps(r)+'\n')
(out/'complete.json').write_text(json.dumps(results,indent=2))

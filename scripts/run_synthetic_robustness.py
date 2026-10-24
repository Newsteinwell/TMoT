"""Calibrate ER64 on the old development stream, then freeze all choices."""
import concurrent.futures,json,os,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);assert shutil.disk_usage(root).free>12*1024**3
folder=root/'runs/synthetic_robustness_v55';folder.mkdir(exist_ok=False);cal=root/'runs/er64_calibration_v55';cal.mkdir(exist_ok=False)
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
def command(out,d,e,v,seed=42,order=17,ridge=None):
 cmd=[sys.executable,'-m','tmot.experiment_synthetic_robustness','--root',str(root),'--out',str(out),'--dataset',d,'--encoder',e,'--variant',v,'--seed',str(seed),'--order',str(order)]
 return cmd+(['--ridge',str(ridge)] if ridge is not None else [])
for e in ('mantis','summary'):
 with (cal/(e+'.log')).open('w') as log:subprocess.run(command(cal,'phonemespectra',e,'er64'),env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=14400)
selected={}
for row in json.loads((root/'synthetic_capacity_v54_comparison.json').read_text())['selected']:
 if row['variant'] in ('task_n256','er'):
  selected[row['dataset']+'_'+row['encoder']+'_'+('task' if row['variant']=='task_n256' else 'er4')]=row['ridge']
for e in ('mantis','summary'):
 summaries=[json.loads(p.read_text()) for p in cal.glob(f'phonemespectra_{e}_er64_o17_s42_r*/summary.json')];assert len(summaries)==6
 selected['phonemespectra_'+e+'_er64']=min(summaries,key=lambda s:(-s['final_macro_accuracy'],s['config']['ridge']))['config']['ridge']
(folder/'frozen.json').write_text(json.dumps(dict(selected=selected,seeds=[42,43,44],class_orders=[17,23],calibration='ER64 uses only original seed42/order17; no reselection elsewhere',source=os.environ['TMOT_SOURCE_COMMIT']),indent=2))
jobs=[]
for d in ('phonemespectra','lsst','handwriting','energy','beijing_pm25','air_quality'):
 for e in ('mantis','summary'):
  for seed in (42,43,44):
   for order in ((17,23) if d in ('phonemespectra','lsst','handwriting') else (17,)):
    for v in (('task','er4','er64') if d=='phonemespectra' else ('task','er4')):
     ridge=selected[d+'_'+e+'_'+v];name=f'{d}_{e}_{v}_o{order}_s{seed}_r{ridge:g}'
     jobs.append(dict(name=name,command=command(folder,d,e,v,seed,order,ridge)))
assert len(jobs)==120
(folder/'manifest.json').write_text(json.dumps(dict(jobs=jobs,pid=os.getpid(),workers=20,source=os.environ['TMOT_SOURCE_COMMIT']),indent=2))
def run(job):
 with (folder/(job['name']+'.log')).open('w') as log:
  try:p=subprocess.run(job['command'],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=14400);status='completed' if p.returncode==0 else 'failed'
  except subprocess.TimeoutExpired:status='timeout'
 return dict(**job,status=status)
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
 for item in pool.map(run,jobs):
  results.append(item)
  with (folder/'status.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
  print(len(results),item['name'],item['status'],flush=True)
(folder/'complete.json').write_text(json.dumps(results,indent=2))

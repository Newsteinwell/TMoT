import concurrent.futures,json,os,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);folder=root/'joint_readout_v50';folder.mkdir(exist_ok=False);jobs=[(d,e) for d in ('phonemespectra','lsst','handwriting','energy','beijing_pm25','air_quality') for e in ('mantis','summary')]
(folder/'manifest.json').write_text(json.dumps(dict(jobs=jobs,pid=os.getpid(),workers=12,source=os.environ['TMOT_SOURCE_COMMIT']),indent=2))
def run(job):
 d,e=job;name=d+'_'+e
 with (folder/(name+'.log')).open('w') as f:
  result=subprocess.run([sys.executable,'-m','scripts.diagnose_joint_readout',str(root),d,e,str(folder/name)],stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
 return dict(dataset=d,encoder=e,status='completed' if result.returncode==0 else 'failed')
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:results=list(pool.map(run,jobs))
(folder/'complete.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))

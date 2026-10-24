import concurrent.futures,json,os,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);out=root/'runs/margin_selector_v66';out.mkdir(exist_ok=False)
source=root/'runs/bounded_robustness_v64';terminal=json.loads((source/'complete.json').read_text());assert len(terminal)==36 and all(s['status']=='completed' for s in terminal)
groups=sorted(p.stem for p in (source/'_shared').glob('*.pt'));assert len(groups)==36
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
(out/'manifest.json').write_text(json.dumps(dict(groups=groups,source=os.environ['TMOT_SOURCE_COMMIT'],workers=6,pid=os.getpid()),indent=2))
def run(group):
 with (out/(group+'.log')).open('x') as log:
  p=subprocess.run([sys.executable,'-m','scripts.diagnose_margin_selector','--root',str(root),'--out',str(out),'--group',group],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
 return dict(group=group,status='completed' if p.returncode==0 else 'failed',returncode=p.returncode)
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 for f in concurrent.futures.as_completed([pool.submit(run,g) for g in groups]):
  r=f.result();results.append(r);print(r,flush=True)
  with (out/'status.jsonl').open('a') as file:file.write(json.dumps(r)+'\n')
(out/'complete.json').write_text(json.dumps(results,indent=2))

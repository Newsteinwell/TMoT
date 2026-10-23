"""Bounded validation follow-ups: DIL core audit and low-rank semantic completion."""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--wait-pid',type=int,default=0);a=p.parse_args()
if a.wait_pid:
    proc=Path(f'/proc/{a.wait_pid}/cmdline')
    while proc.exists() and b'run_transport_queue.py' in proc.read_bytes():time.sleep(10)
out=a.root/'runs'/'followups_v4';out.mkdir(parents=True,exist_ok=True)
variants={'full':[],'ridge':['--readout','ridge'],'frozen':['--frozen'],'no_hist':['--hist-weight','0']}
jobs=[('dil','har',v,s) for v in variants for s in (42,43)]
jobs += [('residual','uwave',v,s) for v in ('full','no_amplitude') for s in (42,43)]
variants['no_amplitude']=['--no-amplitude']
(out/'manifest.json').write_text(json.dumps(dict(jobs=jobs,pid=os.getpid(),source=os.environ['TMOT_SOURCE_COMMIT'],workers=8),indent=2))
def run(job):
    protocol,d,v,seed=job;name=f'{d}_{protocol}_{v}_o17_s{seed}';dest=out/name
    if dest.exists():return dict(name=name,status='existing_not_retried')
    cmd=[sys.executable,'-m',f'tmot.experiment_{protocol}','--root',str(a.root),'--out',str(dest),
         '--dataset',d,'--seed',str(seed),'--order','17','--eval-split','val']+variants[v]
    start=time.time()
    with (out/(name+'.log')).open('w') as f:
        try:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=14400);status='completed' if r.returncode==0 else 'failed'
        except subprocess.TimeoutExpired:status='timeout'
    return dict(name=name,status=status,seconds=time.time()-start,command=cmd)
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    for result in pool.map(run,jobs):
        results.append(result);print(json.dumps(result),flush=True)
        with (out/'status.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
(out/'complete.json').write_text(json.dumps(results,indent=2))

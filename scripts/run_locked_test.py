"""Locked test configurations selected before reading round-two test results."""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--wait-pid',type=int,default=0);a=p.parse_args()
if a.wait_pid:
    proc=Path(f'/proc/{a.wait_pid}/cmdline')
    while proc.exists() and b'run_followups.py' in proc.read_bytes():time.sleep(10)
out=a.root/'runs'/'locked_test_v5';out.mkdir(parents=True,exist_ok=True)
jobs=[]
for d in ('har','uwave','dailysports'):
    for variant,module,args in [('full','transport',[]),('ridge','transport',['--readout','ridge']),
                                ('matched_ridge','matched',['--method','ridge']),('matched_slda','matched',['--method','slda'])]:
        for seed in (42,43):jobs.append((d,variant,module,args,seed))
for seed in (42,43):jobs.append(('uwave','no_amplitude','transport',['--no-amplitude'],seed))
for v,module,args in [('full_r1','transport',['--ridge','1']),('no_hist_r1','transport',['--hist-weight','0','--ridge','1']),
                     ('frozen_r01','transport',['--frozen','--ridge','.1']),('matched_ridge_r1','matched',['--method','ridge','--ridge','1'])]:
    for seed in (42,43):jobs.append(('energy',v,module,args,seed))
(out/'manifest.json').write_text(json.dumps(dict(jobs=jobs,pid=os.getpid(),source=os.environ['TMOT_SOURCE_COMMIT'],workers=8),indent=2))
def run(job):
    d,v,module,args,seed=job;name=f'{d}_{v}_o17_s{seed}';dest=out/name
    if dest.exists():return dict(name=name,status='existing_not_retried')
    cmd=[sys.executable,'-m',f'tmot.experiment_{module}','--root',str(a.root),'--out',str(dest),
         '--dataset',d,'--seed',str(seed),'--order','17','--eval-split','test']+args
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

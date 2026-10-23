"""Use up to sixteen total training slots without restarting existing jobs."""
import argparse,concurrent.futures,json,os,subprocess,sys,time,threading
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--existing-pid',type=int,default=8210);a=p.parse_args()
out=a.root/'runs'/'locked_extension_v6';out.mkdir(parents=True,exist_ok=True)
jobs=[]
for d in ('har','uwave','dailysports'):
    for v,m,args in [('full','transport',[]),('ridge','transport',['--readout','ridge']),
                     ('matched_ridge','matched',['--method','ridge']),('matched_slda','matched',['--method','slda'])]:
        for s in (42,43):jobs.append((d,v,m,args,29,s))
for s in (42,43):jobs.append(('uwave','no_amplitude','transport',['--no-amplitude'],29,s))
for order in (17,29):
    for s in (42,43):jobs.append(('uwave','rank_completed','residual',[],order,s))
(out/f'manifest_{os.getpid()}.json').write_text(json.dumps(dict(jobs=jobs,pid=os.getpid(),source=os.environ['TMOT_SOURCE_COMMIT'],total_concurrency=16,selection='v2/v3/v4 validation only; no v5 test inspection'),indent=2))
lock=threading.Lock();active=0

def capacity():
    external=0
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            cmd=proc.joinpath('cmdline').read_bytes()
            if b'-m\0tmot.experiment' not in cmd:continue
            status=proc.joinpath('status').read_text()
            parent=int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:')))
            if parent!=os.getpid():external+=1
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return max(0,16-external)

def run(job):
    global active
    d,v,module,args,order,seed=job;name=f'{d}_{v}_o{order}_s{seed}';dest=out/name
    if dest.exists():return dict(name=name,status='existing_not_retried')
    while True:
        with lock:
            if active<capacity():active+=1;break
        time.sleep(.5)
    cmd=[sys.executable,'-m',f'tmot.experiment_{module}','--root',str(a.root),'--out',str(dest),
         '--dataset',d,'--seed',str(seed),'--order',str(order),'--eval-split','test']+args
    start=time.time()
    try:
        with (out/(name+'.log')).open('w') as f:
            try:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=14400);status='completed' if r.returncode==0 else 'failed'
            except subprocess.TimeoutExpired:status='timeout'
    finally:
        with lock:active-=1
    return dict(name=name,status=status,seconds=time.time()-start,command=cmd)
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    futures=[pool.submit(run,j) for j in jobs]
    for future in concurrent.futures.as_completed(futures):
        result=future.result();results.append(result);print(json.dumps(result),flush=True)
        with (out/f'status_{os.getpid()}.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
(out/f'complete_{os.getpid()}.json').write_text(json.dumps(results,indent=2))

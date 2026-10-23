"""Resumable bounded pilot queue. Completed jobs skipped, failed jobs never auto-retried."""
import argparse,concurrent.futures,datetime,json,os,subprocess,sys,time
from pathlib import Path


def run_job(job,root,encoder,queue_root,epochs):
    dataset,method,order,seed=job;name=f'{dataset}_{method}_o{order}_s{seed}'
    out=queue_root/name;log=queue_root/(name+'.log')
    if (out/'summary.json').exists():return dict(job=name,status='already_completed')
    if out.exists():return dict(job=name,status='existing_incomplete_not_retried')
    cmd=[sys.executable,'-m','tmot.experiment','--root',str(root),'--out',str(out),
         '--encoder',encoder,'--dataset',dataset,'--method',method,'--order',str(order),'--seed',str(seed),
         '--epochs',str(epochs)]
    start=time.time();env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
    with log.open('w') as f:
        try:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,timeout=14400);status='completed' if r.returncode==0 else 'failed';code=r.returncode
        except subprocess.TimeoutExpired:status='timeout';code=-1
    return dict(job=name,status=status,exit_code=code,elapsed_s=time.time()-start,command=cmd)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    cfg=json.loads(a.config.read_text());out=a.root/'runs'/a.name;out.mkdir(parents=True,exist_ok=True)
    jobs=[(d,m,o,s) for d in cfg['datasets'] for m in cfg['methods'] for o in cfg['orders'] for s in cfg['seeds']]
    (out/'queue_manifest.json').write_text(json.dumps(dict(config=cfg,jobs=jobs,pid=os.getpid(),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run_job,j,a.root,cfg['encoder'],out,cfg['epochs']) for j in jobs]
        for fut in concurrent.futures.as_completed(futures):
            r=fut.result();results.append(r);print(json.dumps(r),flush=True)
            with (out/'queue_status.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    (out/'queue_complete.json').write_text(json.dumps(results,indent=2))

"""Fixed 100-run VAL projection diagnostic; does not change encoders or splits."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);out=root/'runs/history_robustness_v43';out.mkdir(parents=True,exist_ok=False)
selection=json.loads(Path('configs/history_robustness_v43.json').read_text());jobs=[]
for dataset,methods in selection['selected'].items():
    for order in (selection['classification_orders'] if dataset!='energy' else [17]):
        for seed in selection['seeds']:
            for mode,item in methods.items():
                ridge=item['ridge'];name=f'{dataset}_mantis_{mode}_r{ridge:g}_o{order}_s{seed}'
                cmd=[sys.executable,'-m','tmot.experiment_regularized_innovation','--root',str(root),'--out',str(out/name),
                     '--dataset',dataset,'--encoder','mantis','--method',mode,'--dimension','512','--regularized',
                     '--ridge',str(ridge),'--eval-split','val','--order',str(order),'--seed',str(seed),'--device','cuda','--threads','1']
                jobs.append(dict(name=name,command=cmd))
(out/'manifest.json').write_text(json.dumps(dict(jobs=jobs,workers=16,pid=os.getpid(),source=os.environ['TMOT_SOURCE_COMMIT'],selection=selection),indent=2))
def run(job):
    start=time.monotonic();env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (out/(job['name']+'.log')).open('w') as log:
        try:
            r=subprocess.run(job['command'],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=14400)
            status='completed' if r.returncode==0 else 'failed'
        except subprocess.TimeoutExpired:status='timeout'
    return dict(job,status=status,seconds=time.monotonic()-start)
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    for future in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
        r=future.result();results.append(r);print(json.dumps(r),flush=True)
        with (out/'status.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
(out/'complete.json').write_text(json.dumps(results,indent=2))

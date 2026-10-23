"""Predeclared 72-run equal-update nonlinear mechanism screen on existing development VAL."""
import concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);out=root/'runs'/'local_warm_v32';out.mkdir(parents=True,exist_ok=False)
jobs=[]
selection=json.loads(Path('configs/round3/nonlinear_v12_base_selection.json').read_text())
for dataset in ('natops','epilepsy','racketsports','pendigits','articulary','beijing_pm25','air_quality'):
    for encoder in (('mantis','summary') if dataset in ('beijing_pm25','air_quality') else ('mantis',)):
        for loss in ('mse','task'):
            for components,warm in [(k,w) for k in (1,4) for w in (False,True)]:
                mode='local'
                weight=.1
                name=f'{dataset}_{encoder}_{loss}_k{components}_warm{int(warm)}_o17_s42'
                cmd=[sys.executable,'-m','tmot.experiment_nonlinear','--root',str(root),'--out',str(out/name),
                     '--dataset',dataset,'--encoder',encoder,'--method',mode,'--residual-weight',str(weight),
                     '--ridge',str(selection[dataset+'_'+encoder]['ridge']),'--residual-loss',loss,
                     '--seed','42','--order','17','--epochs','30','--eval-split','val','--threads','1','--device','cuda','--er-resample']
                cmd.extend(['--local-components',str(components),'--preserve-targets'])
                if warm:cmd.append('--warm-start')
                jobs.append(dict(name=name,command=cmd))
(out/'manifest.json').write_text(json.dumps(dict(jobs=jobs,workers=16,pid=os.getpid(),source=os.environ['TMOT_SOURCE_COMMIT'],selection='seven new benchmark VAL only'),indent=2))
def run(job):
    start=time.monotonic()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (out/(job['name']+'.log')).open('w') as log:
        try:
            result=subprocess.run(job['command'],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=14400)
            status='completed' if result.returncode==0 else 'failed'
        except subprocess.TimeoutExpired:status='timeout'
    return dict(job,status=status,seconds=time.monotonic()-start)
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
    for future in concurrent.futures.as_completed([pool.submit(run,j) for j in jobs]):
        result=future.result();results.append(result);print(json.dumps(result),flush=True)
        with (out/'status.jsonl').open('a') as log:log.write(json.dumps(result)+'\n')
(out/'complete.json').write_text(json.dumps(results,indent=2))

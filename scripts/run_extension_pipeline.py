"""Persistently extract TRAIN/VAL features, then launch preregistered v45 jobs."""
import concurrent.futures,json,os,shutil,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);out=root/'extension_pipeline';out.mkdir(exist_ok=False)
names=('phonemespectra','lsst','handwriting');jobs=[]
assert shutil.disk_usage(root).free>15*1024**3,'Less than 15 GiB free before extension'
for name in names:
    assert (root/'prepared'/f'{name}_manifest.json').is_file()
    for encoder in ('mantis','summary'):
        cmd=[sys.executable,'-m','scripts.extract_features','--root',str(root),'--datasets',name,'--encoder',encoder,'--splits','train','val','--device','cuda' if encoder=='mantis' else 'cpu','--channel-batch','512']
        if encoder=='mantis':cmd+=['--weights',str(root/'models/MantisV2')]
        jobs.append(dict(name=name+'_'+encoder,command=cmd))
(out/'manifest.json').write_text(json.dumps(dict(pid=os.getpid(),source=os.environ['TMOT_SOURCE_COMMIT'],feature_workers=2,jobs=jobs),indent=2))
def run(job):
    with (out/(job['name']+'.log')).open('w') as f:
        p=subprocess.run(job['command'],stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1'})
    row=dict(name=job['name'],returncode=p.returncode);print(json.dumps(row),flush=True);return row
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,jobs))
(out/'feature_complete.json').write_text(json.dumps(results,indent=2))
assert all(r['returncode']==0 for r in results),'Feature failure: inspect logs; no training launched'
assert shutil.disk_usage(root).free>15*1024**3,'Less than 15 GiB free before training'
for name in names:
    for encoder in ('mantis','summary'):
        assert not (root/'features'/encoder/f'{name}_test.npz').exists()
        for split in ('train','val'):assert (root/'features'/encoder/f'{name}_{split}.json').is_file()
with (out/'training.log').open('w') as f:r=subprocess.run([sys.executable,'-m','scripts.run_harder_validation',str(root),'--workers','16'],stdout=f,stderr=subprocess.STDOUT)
(out/'complete.json').write_text(json.dumps(dict(training_returncode=r.returncode,finished=time.time()),indent=2));assert r.returncode==0

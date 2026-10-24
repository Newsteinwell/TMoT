"""Broad validation screening, never official test evaluation.

One deterministic basis/order screen, not independent-seed evidence. All ridge
variants receive the same six-value grid; SLDA has fixed 0.1 shrinkage.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('root', type=Path)
p.add_argument('--workers', type=int, default=16)
a = p.parse_args()
out = a.root / 'runs' / 'harder_validation_v45'
out.mkdir(parents=True, exist_ok=False)
jobs = []
for dataset in ('phonemespectra','lsst','handwriting'):
    for encoder in ('mantis','summary'):
        for variant in ('linear','local','er'):
            for ridge in (.001,.01,.1,1.,10.,100.):
                name=f'{dataset}_{encoder}_{variant}_r{ridge:g}_o17_s42'
                module='tmot.experiment_stable' if variant=='linear' else 'tmot.experiment_regularized_innovation'
                cmd=[sys.executable,'-m',module,'--root',str(a.root),'--out',str(out/name),'--dataset',dataset,'--encoder',encoder,'--ridge',str(ridge),'--dimension','512','--eval-split','val','--device','cpu' if variant=='linear' else 'cuda','--threads','1','--order','17','--seed','42','--method','ridge' if variant=='linear' else variant]
                if variant!='linear':cmd+=['--regularized']
                jobs.append(dict(name=name,command=cmd))

(out / 'manifest.json').write_text(json.dumps(dict(jobs=jobs, workers=a.workers, pid=os.getpid(),
    source=os.environ['TMOT_SOURCE_COMMIT'], selection='VAL only; TEST remains sealed'), indent=2))

def run(job):
    start = time.monotonic()
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (out / (job['name'] + '.log')).open('w') as log:
        try:
            result = subprocess.run(job['command'], stdout=log, stderr=subprocess.STDOUT, env=env, timeout=14400)
            status = 'completed' if result.returncode == 0 else 'failed'
        except subprocess.TimeoutExpired:
            status = 'timeout'
    return dict(job, status=status, seconds=time.monotonic() - start)

results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
    for future in concurrent.futures.as_completed([pool.submit(run, job) for job in jobs]):
        result = future.result()
        results.append(result)
        with (out / 'status.jsonl').open('a') as log:
            log.write(json.dumps(result) + '\n')
        print(json.dumps(result), flush=True)
(out / 'complete.json').write_text(json.dumps(results, indent=2))

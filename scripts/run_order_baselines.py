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
out = a.root / 'runs' / 'order_baselines_v21'
out.mkdir(parents=True, exist_ok=False)
jobs = []
selection=json.loads(Path('configs/round3/order_baselines_v21.json').read_text())
for dataset in ('natops','epilepsy','racketsports','pendigits','articulary'):
    for order in (17,29,41):
        for variant in ('linear','static','slda'):
            ridge=selection[dataset+'_'+variant]['ridge']
            name=f'{dataset}_mantis_{variant}_r{ridge:g}_o{order}_s42'
            module='tmot.experiment_kernel' if variant=='static' else 'tmot.experiment_stable'
            cmd=[sys.executable,'-m',module,'--root',str(a.root),'--out',str(out/name),
                 '--dataset',dataset,'--encoder','mantis','--ridge',str(ridge),
                 '--eval-split','val','--device','cpu','--threads','1','--order',str(order),'--seed','42']
            cmd+=['--no-transport'] if variant=='static' else ['--method','ridge' if variant=='linear' else 'slda']
            jobs.append(dict(name=name,command=cmd))

(out / 'manifest.json').write_text(json.dumps(dict(jobs=jobs, workers=a.workers, pid=os.getpid(),
    source=os.environ['TMOT_SOURCE_COMMIT'], selection='VAL only; TEST remains sealed'), indent=2))

def run(job):
    start = time.monotonic()
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
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

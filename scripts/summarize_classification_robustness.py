"""Descriptive paired summaries: do not treat shared VAL runs as independent data."""
import argparse,json,statistics
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
runs={}
for path in a.runs.glob('*/summary.json'):
    s=json.loads(path.read_text());c=s['config'];key=(c['dataset'],c['order'],c['seed'],c['method'])
    assert key not in runs and c['eval_split']=='val';runs[key]=s
assert len(runs)==150
metrics=('final_macro_accuracy','average_macro_accuracy','forgetting')
rows=[];dataset_rows=[]
for dataset in sorted({k[0] for k in runs}):
    actual_orders={};all_pairs=defaultdict(list)
    for order in (17,29,41):
        canonical=None;values={m:defaultdict(list) for m in ('gaussian','er')};paired=defaultdict(list)
        for seed in (42,43,44,45,46):
            g=runs[dataset,order,seed,'gaussian'];e=runs[dataset,order,seed,'er']
            assert g['order']==e['order']
            sequence=tuple(tuple(sorted(chunk)) for chunk in g['order'])
            if canonical is not None:assert sequence==canonical
            canonical=sequence
            for metric in metrics:
                values['gaussian'][metric].append(g[metric]);values['er'][metric].append(e[metric])
                paired[metric].append(g[metric]-e[metric]);all_pairs[metric].append(g[metric]-e[metric])
        actual_orders[order]=canonical
        rows.append(dict(dataset=dataset,order_seed=order,actual_class_chunks=canonical,
                         methods={m:{k:dict(mean=statistics.mean(v),seed_sd=statistics.stdev(v)) for k,v in x.items()} for m,x in values.items()},
                         gaussian_minus_er={k:dict(mean=statistics.mean(v),seed_sd=statistics.stdev(v),min=min(v),max=max(v)) for k,v in paired.items()}))
    dataset_rows.append(dict(dataset=dataset,distinct_unordered_arrival_sequences=len(set(actual_orders.values())),
                             paired_differences={k:dict(mean=statistics.mean(v),min=min(v),max=max(v)) for k,v in all_pairs.items()}))
a.out.write_text(json.dumps(dict(scope='VAL descriptive paired effects; no independent-run significance claims',completed=150,orders=rows,datasets=dataset_rows),indent=2))
for row in dataset_rows:print(json.dumps(row))

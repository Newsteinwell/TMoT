"""Compare fixed-parameter class-order VAL runs; descriptive, not blind testing."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('nonlinear',type=Path);p.add_argument('baselines',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
read=lambda p:[json.loads(f.read_text()) for f in p.glob('*/summary.json')]
runs=read(a.nonlinear);bases=read(a.baselines);assert len(runs)==150 and len(bases)==45
rows=[]
for dataset in ('natops','epilepsy','racketsports','pendigits','articulary'):
    metrics={};by_order={}
    for method in ('gaussian','er','linear','slda','static'):
        source=runs if method in ('gaussian','er') else bases
        selected=[]
        for s in source:
            c=s['config']
            if c['dataset']!=dataset:continue
            actual=c.get('method','static');actual='linear' if actual=='ridge' else actual
            if actual==method:selected.append(s)
        assert len(selected)==(15 if method in ('gaussian','er') else 3)
        metrics[method]={m:statistics.mean(s[m] for s in selected) for m in ('final_macro_accuracy','average_macro_accuracy','forgetting')}
        by_order[method]={o:statistics.mean(s['final_macro_accuracy'] for s in selected if s['config']['order']==o) for o in (17,29,41)}
    strongest=max(('er','linear','slda','static'),key=lambda m:metrics[m]['final_macro_accuracy'])
    rows.append(dict(dataset=dataset,means=metrics,final_by_order=by_order,strongest_observed_reference=strongest,
                     gaussian_final_gap_pp=100*(metrics['gaussian']['final_macro_accuracy']-metrics[strongest]['final_macro_accuracy'])))
a.out.write_text(json.dumps(dict(scope='Shared VAL, fixed parameters, means across three orders; stochastic methods five seeds per order; no independent-data significance claim',datasets=rows),indent=2))
for r in rows:print(r['dataset'],r['strongest_observed_reference'],r['gaussian_final_gap_pp'])

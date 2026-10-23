import argparse,json,statistics
from pathlib import Path
from collections import defaultdict
p=argparse.ArgumentParser();p.add_argument('runs',type=Path);p.add_argument('out',type=Path);a=p.parse_args();groups=defaultdict(list)
manifest=json.loads((a.runs/'manifest.json').read_text())
for job in manifest['jobs']:
    s=json.loads((a.runs/(job['name']+'.json')).read_text())
    for r in s['rows']:groups[s['dataset'],s['encoder'],s['anchor'],s['loss'],r['method'],r['count']].append(r)
assert len(groups)==288 and all(len(v)==5 for v in groups.values())
rows=[]
for key,r in sorted(groups.items()):
    cosine=[x['cosine'] for x in r if x['cosine'] is not None];error=[x['relative_error'] for x in r if x['relative_error'] is not None]
    rows.append(dict(dataset=key[0],encoder=key[1],anchor=key[2],loss=key[3],method=key[4],count=key[5],median_cosine=statistics.median(cosine) if cosine else None,median_relative_error=statistics.median(error) if error else None,opposed_draws=sum(x['negative_step_directional_derivative']>0 for x in r),reference_norm=r[0]['reference_norm'],near_zero=r[0]['reference_near_zero']))
a.out.write_text(json.dumps(dict(scope='five draws per fixed anchor; no validation performance inference',rows=rows),indent=2))

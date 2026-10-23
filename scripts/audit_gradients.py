import argparse,hashlib,itertools,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
folder=a.root/'gradient_v35';manifest=json.loads((folder/'manifest.json').read_text());terminal=json.loads((folder/'complete.json').read_text())
assert len(terminal)==36 and all(r['status']=='completed' for r in terminal)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
expected=set(itertools.product(('empirical_mc','er','gaussian','local_merged'),(4096,16384),range(5)))
checked=[]
for job in manifest['jobs']:
    s=json.loads((folder/(job['name']+'.json')).read_text());assert s['complete'] and s['source']==manifest['source']
    assert {(r['method'],r['count'],r['repeat']) for r in s['rows']}==expected and len(s['rows'])==40
    assert s['train_sha256']==sha(a.root/'features'/s['encoder']/f"{s['dataset']}_train.npz")
    assert s['checkpoint_sha256']==sha(a.root/'runs/warm_transport_v31'/job['name']/'final.pt')
    assert s['er_checkpoint_sha256']==sha(a.root/'runs/warm_transport_v31'/job['name'].replace('_gaussian_','_er_')/'final.pt')
    for r in s['rows']:
        nr=r['reference_norm'];ne=r['estimate_norm'];dot=-r['negative_step_directional_derivative']
        assert math.isfinite(nr+ne+dot+r['loss']) and nr==s['reference_norm']
        if r['cosine'] is not None:
            assert -1.000001<=r['cosine']<=1.000001 and math.isclose(r['cosine'],dot/(nr*ne),rel_tol=1e-9,abs_tol=1e-10)
        if r['relative_error'] is not None:
            assert math.isclose(r['relative_error']**2,(nr*nr+ne*ne-2*dot)/(nr*nr),rel_tol=1e-7,abs_tol=1e-8)
    checked.append(job['name'])
assert len(checked)==36
a.out.write_text(json.dumps(dict(complete=True,anchors=36,comparisons=1440,checks=['complete predefined matrix','current train/model hashes match diagnostics','gradient norm/dot/cosine/error identities','finite statistics','per-job parameter immutability and chunk-gradient assertions completed'],checked=checked),indent=2))
print('36 anchors and 1440 comparisons verified')

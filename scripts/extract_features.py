"""Frozen MantisV2 per-channel features, preserving channel order.

Extraction is sample-local and label-free. Evaluation cache stays outside learner.
No PCA or normalization is fitted on future stages. Standardization is in runner.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--datasets',nargs='+',required=True)
    p.add_argument('--encoder',choices=['mantis','summary'],default='mantis');p.add_argument('--weights',type=Path)
    p.add_argument('--channel-batch',type=int,default=128);p.add_argument('--device',default='cuda');a=p.parse_args()
    torch.set_num_threads(4)
    model=None
    if a.encoder=='mantis':
        from mantis.architecture import MantisV2
        model=MantisV2(device=a.device,return_transf_layer=2,output_token='combined',pre_training=False)
        model=model.from_pretrained(str(a.weights)).eval().requires_grad_(False)
    out=a.root/'features'/a.encoder;out.mkdir(parents=True,exist_ok=True)
    model_hash={p.name:sha(p) for p in a.weights.glob('*') if p.is_file()} if a.weights else {}
    for name in a.datasets:
        for split in ('train','val','test'):
            source=a.root/'prepared'/f'{name}_{split}.npz';target=out/f'{name}_{split}.npz'
            if target.exists():raise FileExistsError(f'Explicitly remove stale cache before replacing: {target}')
            raw=np.load(source);x=raw['x'];start=time.perf_counter();rows=[]
            if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                for i in range(0,len(x),32):
                    bx=torch.as_tensor(x[i:i+32],device=a.device)
                    # Fixed per-series centering/scaling for semantic encoder; amplitudes retained separately.
                    means=bx.mean(-1);std=bx.std(-1,unbiased=False).clamp_min(1e-6)
                    norm=(bx-means[:,:,None])/std[:,:,None]
                    if model is None:
                        seq=F.interpolate(norm,size=32,mode='linear',align_corners=False).cpu().numpy().reshape(len(bx),-1)
                    else:
                        series=F.interpolate(norm,size=512,mode='linear',align_corners=False).reshape(-1,1,512)
                        embeds=[]
                        for j in range(0,len(series),a.channel_batch):
                            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.device.startswith('cuda')):
                                embeds.append(model(series[j:j+a.channel_batch]).float().cpu().numpy())
                        seq=np.concatenate(embeds).reshape(len(bx),-1)
                    rows.append(np.concatenate([seq,means.cpu().numpy(),std.cpu().numpy()],axis=1))
                    if i%512==0:print(name,split,i,'/',len(x),flush=True)
            features=np.concatenate(rows).astype('float32');assert np.isfinite(features).all()
            np.savez_compressed(target,x=features,y=raw['y'],ids=raw['ids'],groups=raw['groups'])
            meta=dict(dataset=name,split=split,encoder=a.encoder,seconds=time.perf_counter()-start,
                 shape=list(features.shape),source_sha256=sha(source),cache_sha256=sha(target),weights=model_hash,
                 gpu_peak_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                 preprocessing='sample-local zscore to encoder; ordered channel concat + channel mean/population-SD')
            target.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(meta,flush=True)

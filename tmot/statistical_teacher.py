"""Convert distances to a finite, scale-invariant all-seen-class target."""
import numpy as np

def distance_probabilities(distances):
 d=np.asarray(distances,dtype=np.float64)
 if d.ndim!=2 or not d.shape[1] or not np.isfinite(d).all() or np.any(d<0):raise ValueError('Expected nonnegative finite class distances')
 scale=np.maximum(d.mean(1,keepdims=True),1e-12)
 logits=-d/scale;logits-=logits.max(1,keepdims=True)
 p=np.exp(logits);return p/p.sum(1,keepdims=True)


def teacher_targets(distances,mode='soft'):
 """Fixed confidence ablation; these targets are not calibrated posteriors."""
 d=np.asarray(distances,dtype=np.float64)
 p=distance_probabilities(d)
 if mode=='soft':return p
 if mode=='hard':
  out=np.zeros_like(p);out[np.arange(len(d)),d.argmin(1)]=1.;return out
 if mode!='sharp':raise ValueError(mode)
 # Temperature 0.1 relative to the original row-mean distance scale.
 logits=-d/(.1*np.maximum(d.mean(1,keepdims=True),1e-12))
 logits-=logits.max(1,keepdims=True);out=np.exp(logits);return out/out.sum(1,keepdims=True)

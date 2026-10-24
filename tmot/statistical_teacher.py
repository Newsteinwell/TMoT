"""Convert distances to a finite, scale-invariant all-seen-class target."""
import numpy as np

def distance_probabilities(distances):
 d=np.asarray(distances,dtype=np.float64)
 if d.ndim!=2 or not d.shape[1] or not np.isfinite(d).all() or np.any(d<0):raise ValueError('Expected nonnegative finite class distances')
 scale=np.maximum(d.mean(1,keepdims=True),1e-12)
 logits=-d/scale;logits-=logits.max(1,keepdims=True)
 p=np.exp(logits);return p/p.sum(1,keepdims=True)

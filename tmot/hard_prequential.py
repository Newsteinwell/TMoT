"""Fixed-order tie-breaking for minimum cumulative pre-task error."""
import numpy as np

def hard_weights(sse,count):
 sse=np.asarray(sse,np.float64)
 if sse.ndim!=1 or not len(sse) or not np.isfinite(sse).all() or (sse<0).any() or count<0:raise ValueError('Invalid cumulative errors')
 if count==0:return np.ones(len(sse))/len(sse)
 w=np.zeros(len(sse));w[np.argmin(sse)]=1.
 return w

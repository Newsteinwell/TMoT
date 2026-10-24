"""Parameter-free relative distance margin; no labels or fitted calibration."""
import numpy as np

def relative_margin(distances):
 d=np.asarray(distances,np.float64)
 if d.ndim!=2 or d.shape[1]<2 or not np.isfinite(d).all() or np.any(d<0):
  raise ValueError('Expected finite nonnegative distances for at least two classes')
 first=np.partition(d,1,axis=1)[:,:2];low=np.min(first,axis=1);high=np.max(first,axis=1)
 return np.divide(high-low,high+low,out=np.zeros(len(d)),where=(high+low)>0)

def select_by_margin(reference,candidate,classes):
 reference=np.asarray(reference);candidate=np.asarray(candidate);classes=np.asarray(classes)
 if reference.shape!=candidate.shape or reference.shape[1]!=len(classes):raise ValueError('Class coordinates must agree')
 choose=relative_margin(candidate)>relative_margin(reference)
 prediction=np.where(choose,candidate.argmin(1),reference.argmin(1))
 return classes[prediction],choose

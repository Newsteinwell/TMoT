"""Real-feature neighborhood diagnostic, explicitly not statistics-only TMoT."""
import numpy as np


class NeighborReadout:
    def __init__(self, dimension, task, neighbors=5, budget=0, seed=42):
        self.dimension=dimension;self.task=task;self.neighbors=neighbors
        self.budget=budget;self.rng=np.random.default_rng(seed)
        self.x=None;self.y=None;self.seen=0;self.last_loss=None

    def observe(self,x,y):
        x=np.asarray(x,dtype=np.float32);y=np.asarray(y)
        if self.x is None:
            self.capacity=(self.budget//(x.shape[1]*4+y[0].nbytes)) if self.budget else None
            if self.capacity==0:raise ValueError('budget cannot hold one feature row')
            self.x=np.empty((0,x.shape[1]),dtype=np.float32);self.y=np.empty((0,)+y.shape[1:],dtype=y.dtype)
        for row,target in zip(x,y):
            self.seen+=1
            if self.capacity is None or len(self.x)<self.capacity:
                self.x=np.concatenate((self.x,row[None]));self.y=np.concatenate((self.y,np.asarray(target)[None]))
            else:
                j=self.rng.integers(self.seen)
                if j<self.capacity:self.x[j]=row;self.y[j]=target

    def predict(self,x):
        result=[];k=min(self.neighbors,len(self.x))
        train=self.x.astype(np.float64);norm=np.sum(train**2,axis=1)
        for batch in np.array_split(np.asarray(x,dtype=np.float64),max(1,(len(x)+127)//128)):
            d=np.maximum(np.sum(batch**2,axis=1)[:,None]+norm[None]-2*batch@train.T,0)
            # Stable distance ties use retained row order; vote ties use smallest class.
            ids=np.argsort(d,axis=1,kind='stable')[:,:k];targets=self.y[ids]
            if self.task=='classification':
                classes=np.unique(self.y)
                votes=(targets[:,:,None]==classes).sum(axis=1)
                result.append(classes[votes.argmax(axis=1)])
            else:result.append(targets.mean(axis=1))
        return np.concatenate(result)

    @property
    def persistent_bytes(self):
        return 0 if self.x is None else self.x.nbytes+self.y.nbytes

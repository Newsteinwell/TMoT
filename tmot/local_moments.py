"""Low-rank local moment mixtures for approximate nonlinear history replay.

Stores component counts, means, covariance factors and diagonal residuals.
Components are fitted only on the arriving batch; no historical row buffer.
The number of components can grow with arrivals; this is not fixed-size memory.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class LocalCell:
    n: int
    mean: np.ndarray
    factor: np.ndarray
    residual_sd: np.ndarray

    @classmethod
    def fit(cls,rows,rank=16,shrinkage=.1):
        rows=np.asarray(rows,dtype=np.float64);mean=rows.mean(0);center=rows-mean
        sd=np.maximum(np.sqrt(np.mean(center**2,axis=0)),1e-8)
        normalized=center/sd
        _,values,vectors=np.linalg.svd(normalized,full_matrices=False)
        r=min(rank,max(len(rows)-1,0),len(values))
        factor=(vectors[:r].T*values[:r]/np.sqrt(len(rows)))*sd[:,None]*np.sqrt(1-shrinkage)
        residual=np.maximum(np.mean(center**2,axis=0)-np.sum(factor**2,axis=1),0)
        return cls(len(rows),mean.astype(np.float32),factor.astype(np.float32),np.sqrt(residual).astype(np.float32))

    @classmethod
    def fit_target_preserving(cls,rows,input_dim,rank=16,shrinkage=.1):
        """Protect empirical XY and YY covariance, compress only X residual.

        SVD of centered targets avoids inverting a singular YY matrix. If U
        spans the observed target column space, X = UU'X + (I-UU')X.
        The first part and Y share latent draws; only the orthogonal residual
        is truncated/shrunk. XY, YY and diagonal XX are preserved to rounding
        tolerance (target singular values below 1e-12 relative are discarded).
        """
        rows=np.asarray(rows,dtype=np.float64);mean=rows.mean(0);center=rows-mean
        x,y=center[:,:input_dim],center[:,input_dim:]
        if not y.shape[1]:raise ValueError('At least one target required')
        u,values,vectors=np.linalg.svd(y,full_matrices=False)
        keep=values>max(float(values.max(initial=0))*1e-12,1e-15)
        u,values,vectors=u[:,keep],values[keep],vectors[keep]
        target_x=x.T@u/np.sqrt(len(x))
        target_y=vectors.T*values/np.sqrt(len(x))
        residual=x-u@(u.T@x)
        compressed=cls.fit(residual,rank,shrinkage)
        factor=np.zeros((rows.shape[1],len(values)+compressed.factor.shape[1]))
        factor[:input_dim,:len(values)]=target_x
        factor[input_dim:,:len(values)]=target_y
        factor[:input_dim,len(values):]=compressed.factor
        diagonal=np.r_[compressed.residual_sd,np.zeros(y.shape[1])]
        return cls(len(rows),mean.astype(np.float32),factor.astype(np.float32),diagonal.astype(np.float32))

    def sample(self,n,rng):
        return (self.mean+rng.normal(size=(n,self.factor.shape[1]))@self.factor.T+
                rng.normal(size=(n,len(self.mean)))*self.residual_sd).astype(np.float32)

    @property
    def nbytes(self):return self.mean.nbytes+self.factor.nbytes+self.residual_sd.nbytes+8


def local_partition(x,k,rng):
    x=np.asarray(x,dtype=np.float64)
    mean=x.mean(0);sd=np.maximum(x.std(0),1e-6)
    projection=rng.normal(size=(x.shape[1],min(16,x.shape[1])))/np.sqrt(x.shape[1])
    z=((x-mean)/sd)@projection
    centers=[z[rng.integers(len(z))]]
    while len(centers)<k:
        distance=np.min(((z[:,None]-np.array(centers)[None])**2).sum(2),axis=1)
        if distance.sum()<1e-12:break
        centers.append(z[rng.choice(len(z),p=distance/distance.sum())])
    centers=np.array(centers)
    for _ in range(20):
        assignment=((z[:,None]-centers[None])**2).sum(2).argmin(1)
        updated=np.array([z[assignment==j].mean(0) if np.any(assignment==j) else c for j,c in enumerate(centers)])
        if np.allclose(updated,centers):break
        centers=updated
    return ((z[:,None]-centers[None])**2).sum(2).argmin(1)


class LocalMomentMixture:
    def __init__(self,dimension,task,seed=42,components=4,rank=16,preserve_targets=False):
        self.dimension,self.task,self.components,self.rank=dimension,task,components,rank
        self.preserve_targets=preserve_targets
        self.cells={};self.rng=np.random.default_rng(seed)

    def update(self,x,y):
        x=np.asarray(x);y=np.asarray(y)
        labels=np.unique(y) if self.task=='classification' else [None]
        for label in labels:
            mask=y==label if label is not None else np.ones(len(x),dtype=bool)
            inputs=x[mask]
            targets=y[mask]
            if targets.ndim==1:targets=targets[:,None]
            rows=inputs if label is not None else np.concatenate([inputs,targets],axis=1)
            k=min(self.components,max(1,len(inputs)//16))
            assignment=local_partition(inputs,k,self.rng)
            for component in np.unique(assignment):
                batch=rows[assignment==component]
                cell=(LocalCell.fit_target_preserving(batch,self.dimension,self.rank)
                      if self.preserve_targets and self.task=='regression'
                      else LocalCell.fit(batch,self.rank))
                self.cells.setdefault(label,[]).append(cell)

    def sample(self,count,classes,rng):
        labels=list(classes) if self.task=='classification' else [None]
        allocations=rng.multinomial(count,np.ones(len(labels))/len(labels))
        xs=[];ys=[]
        for label,n in zip(labels,allocations):
            cells=self.cells[label];weights=np.array([c.n for c in cells],dtype=float);weights/=weights.sum()
            sizes=rng.multinomial(n,weights)
            for cell,size in zip(cells,sizes):
                if not size:continue
                generated=cell.sample(size,rng)
                xs.append(generated[:,:self.dimension])
                ys.append(np.full(size,label) if label is not None else generated[:,self.dimension:])
        return np.concatenate(xs),np.concatenate(ys)

    @property
    def nbytes(self):return sum(c.nbytes for cells in self.cells.values() for c in cells)

"""Generalized WiMoT first-stage semantic/amplitude/PCA basis.

The input is ordered channel embeddings followed by channel means and SDs.
Only first-stage training data fit normalization, supervised stem and PCA.
The frozen basis is shared by classification and scalar/vector regression.
"""
import copy
import numpy as np
import torch
from torch import nn
from .learner import supcon


class DualStem(nn.Module):
    def __init__(self, semantic_dim, amplitude_dim):
        super().__init__()
        self.semantic_dim=semantic_dim
        self.semantic=nn.Sequential(nn.Linear(semantic_dim,512),nn.BatchNorm1d(512),nn.ReLU(),
                                    nn.Linear(512,128),nn.BatchNorm1d(128),nn.ReLU())
        self.amplitude=nn.Sequential(nn.Linear(amplitude_dim,1024,bias=False),nn.ReLU(),
                                     nn.Linear(1024,128,bias=False))
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.kaiming_normal_(layer.weight,mode='fan_out',nonlinearity='relu')
                if layer.bias is not None:nn.init.zeros_(layer.bias)
        self.amplitude.requires_grad_(False)
        self.register_buffer('amp_mean',torch.zeros(128))
        self.register_buffer('amp_sd',torch.ones(128))

    @torch.no_grad()
    def calibrate(self,x):
        a=self.amplitude(x[:,self.semantic_dim:]);sd=a.std(0,unbiased=True)
        self.amp_mean.copy_(a.mean(0));self.amp_sd.copy_(sd.clamp_min(max(float(sd.mean())*.01,1e-6)))

    def forward(self,x):
        a=(self.amplitude(x[:,self.semantic_dim:])-self.amp_mean)/self.amp_sd
        return torch.cat([self.semantic(x[:,:self.semantic_dim]),a],dim=1)


class DualPCABasis:
    dimension=512
    def __init__(self,amplitude_dim,task='classification',seed=42,epochs=100,device='cuda',
                 use_amplitude=True,use_pca=True):
        self.amplitude_dim,self.task,self.seed,self.epochs,self.device=amplitude_dim,task,seed,epochs,device
        self.use_amplitude,self.use_pca=use_amplitude,use_pca
        self.fitted=False

    def fit(self,x,y):
        if self.fitted:raise RuntimeError('First-stage basis is immutable')
        x=np.asarray(x,dtype=np.float32);y=np.asarray(y)
        if len(x)<2 or x.ndim!=2 or not np.isfinite(x).all():raise ValueError('At least two finite rows required')
        self.semantic_dim=x.shape[1]-self.amplitude_dim
        if self.semantic_dim<=0 or self.amplitude_dim<=0:raise ValueError('Invalid feature partition')
        self.mean=x.mean(0);sd=x.std(0)
        self.sd=np.maximum(sd,max(float(np.median(sd))*.01,1e-5))
        nx=(x-self.mean)/self.sd
        torch.manual_seed(self.seed);rng=np.random.default_rng(self.seed)
        self.stem=DualStem(self.semantic_dim,self.amplitude_dim).to(self.device)
        tx=torch.as_tensor(nx,device=self.device);self.stem.calibrate(tx)
        adapter=nn.Linear(256,128).to(self.device)
        with torch.no_grad():
            adapter.weight.copy_(torch.cat([torch.eye(128),torch.eye(128)],dim=1).to(self.device)/np.sqrt(2))
            adapter.bias.zero_()
        projector=nn.Linear(128,128).to(self.device)
        if self.task=='classification':
            self.classes=sorted(np.unique(y).tolist());ty=torch.tensor([self.classes.index(c) for c in y],device=self.device)
            outputs=len(self.classes)
        else:
            if y.ndim==1:y=y[:,None]
            ty=torch.as_tensor(y,dtype=torch.float32,device=self.device);outputs=y.shape[1]
            self.classes=[]
        head=nn.Linear(128,outputs).to(self.device)
        parameters=list(self.stem.parameters())+list(adapter.parameters())+list(projector.parameters())+list(head.parameters())
        opt=torch.optim.AdamW([p for p in parameters if p.requires_grad],lr=.001,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,max(self.epochs,1))
        losses=[]
        for epoch in range(self.epochs):
            self.stem.train();order=rng.permutation(len(x))
            # Merge a singleton remainder into its previous batch (BatchNorm requires >=2).
            chunks=list(np.array_split(order,max(1,int(np.ceil(len(order)/128)))))
            for ids in chunks:
                phi=self.stem(tx[ids])
                if not self.use_amplitude:phi=torch.cat([phi[:,:128],torch.zeros_like(phi[:,128:])],dim=1)
                h=adapter(phi);score=head(h)
                loss=(nn.functional.cross_entropy(score,ty[ids])+.5*supcon(projector(h),ty[ids],.07)
                      if self.task=='classification' else (score-ty[ids]).square().sum(1).mean())
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite first-stage loss')
                opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
            scheduler.step()
        self.stem.eval().requires_grad_(False)
        # PCA via the sample Gram matrix; rank truncation explicit, never fit on future arrivals.
        z=nx[:,:self.semantic_dim].astype(np.float64);self.pca_mean=z.mean(0);z-=self.pca_mean
        if len(z)<=z.shape[1]:
            vals,vecs=np.linalg.eigh(z@z.T);order=np.argsort(vals)[::-1];vals=vals[order];vecs=vecs[:,order]
            k=min(256,int(np.sum(vals>max(float(vals[0])*1e-6,1e-12))))
            directions=(vecs[:,:k].T@z)/np.sqrt(vals[:k,None])
        else:
            vals,vecs=np.linalg.eigh(z.T@z);order=np.argsort(vals)[::-1];vals=vals[order];vecs=vecs[:,order]
            k=min(256,int(np.sum(vals>max(float(vals[0])*1e-6,1e-12))))
            directions=vecs[:,:k].T
        self.pca=np.zeros((256,self.semantic_dim),dtype=np.float32)
        if k:
            signs=np.sign(directions[np.arange(k),np.argmax(np.abs(directions),axis=1)])
            self.pca[:k]=(directions*signs[:,None]).astype(np.float32)
        projected=z@self.pca.T
        self.pca_scale=max(float(np.sqrt(np.mean(projected**2))),1e-6)
        self.initial_adapter=copy.deepcopy(adapter).cpu()
        self.initial_projector=copy.deepcopy(projector).cpu();self.initial_head=copy.deepcopy(head).cpu()
        self.metadata=dict(first_rows=len(x),pca_rank=k,pca_padded_to=256,
                           pca_captured_energy=float(vals[:k].sum()/max(np.square(z).sum(),1e-30)),
                           final_training_loss=float(np.mean(losses[-len(chunks):])) if losses else None,
                           task=self.task,semantic_dim=self.semantic_dim,amplitude_dim=self.amplitude_dim,
                           use_amplitude=self.use_amplitude,use_pca=self.use_pca)
        self.fitted=True
        return self

    def transform(self,x):
        if not self.fitted:raise RuntimeError('Fit first-stage training data first')
        x=np.asarray(x,dtype=np.float32);nx=(x-self.mean)/self.sd;rows=[]
        with torch.no_grad():
            for start in range(0,len(x),256):
                rows.append(self.stem(torch.as_tensor(nx[start:start+256],device=self.device)).cpu().numpy())
        dual=np.concatenate(rows)
        if not self.use_amplitude:dual[:,128:]=0
        pca=((nx[:,:self.semantic_dim]-self.pca_mean)@self.pca.T/self.pca_scale).astype(np.float32)
        if not self.use_pca:pca[:]=0
        return np.concatenate([dual,pca],axis=1).astype(np.float32)

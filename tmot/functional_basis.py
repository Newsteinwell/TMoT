"""First-arrival supervised functions, then immutable exact-moment coordinates."""
import numpy as np
import torch
from torch import nn

class FunctionalBasis:
    def __init__(self,base,task,mode='trained',seed=42,width=128,epochs=30,device='cuda'):
        self.base=base;self.task=task;self.mode=mode;self.seed=seed
        self.width=width;self.epochs=epochs;self.device=device;self.dimension=base.dimension+width
    def fit(self,x,y,calibration_x=None,sample_weight=None,input_stats=None):
        if hasattr(self,'net'):raise RuntimeError('Functional basis must remain fixed after first arrival')
        z=self.base.transform(x).astype(np.float32);y=np.asarray(y)
        cal=z if calibration_x is None else self.base.transform(calibration_x).astype(np.float32)
        self.mean=cal.mean(0);sd=cal.std(0);self.sd=np.maximum(sd,max(float(np.median(sd))*.01,1e-4))
        if input_stats is not None:self.mean,self.sd=(np.asarray(v,np.float32).copy() for v in input_stats)
        torch.manual_seed(self.seed)
        self.net=nn.Sequential(nn.Linear(self.base.dimension,self.width),nn.ReLU(),nn.Linear(self.width,self.width),nn.ReLU()).to(self.device)
        if self.task=='classification':
            labels,ids=np.unique(y,return_inverse=True);outputs=len(labels);target=torch.as_tensor(ids,device=self.device)
            weights=1/np.bincount(ids)[ids];weights=weights/weights.sum()*len(y)
        else:
            if y.ndim==1:y=y[:,None]
            outputs=y.shape[1];target=torch.as_tensor(y,dtype=torch.float32,device=self.device);weights=np.ones(len(y))
        if sample_weight is not None:
            weights=np.asarray(sample_weight,dtype=np.float64)
            if weights.shape!=(len(y),) or np.any(weights<0) or not np.isfinite(weights).all() or weights.sum()<=0:raise ValueError('Invalid sample weights')
            weights=weights/weights.sum()*len(y)
        head=nn.Linear(self.width,outputs).to(self.device)
        features=torch.as_tensor((z-self.mean)/self.sd,device=self.device)
        weights=torch.as_tensor(weights,dtype=torch.float32,device=self.device)
        self.optimizer_steps=0;self.last_loss=None
        if self.mode=='trained':
            opt=torch.optim.AdamW(list(self.net.parameters())+list(head.parameters()),lr=.001,weight_decay=1e-4)
            for _ in range(self.epochs):
                for batch in torch.randperm(len(y),device=self.device).split(256):
                    prediction=head(self.net(features[batch]))
                    error=nn.functional.cross_entropy(prediction,target[batch],reduction='none') if self.task=='classification' else (prediction-target[batch]).square().mean(1)
                    loss=(weights[batch]*error).mean();opt.zero_grad();loss.backward();opt.step()
                    self.optimizer_steps+=1;self.last_loss=float(loss.detach())
        elif self.mode!='random':raise ValueError(self.mode)
        self.net=self.net.to('cpu').eval()
        for p in self.net.parameters():p.requires_grad_(False)
        with torch.no_grad():h=self.net(torch.from_numpy((cal-self.mean)/self.sd)).numpy()
        if sample_weight is None:
            self.hidden_mean=h.mean(0);self.hidden_scale=max(float(np.sqrt(np.mean((h-self.hidden_mean)**2))),1e-4)
        else:
            if calibration_x is not None:raise ValueError('Weighted hidden calibration uses training representatives')
            w=np.asarray(sample_weight,dtype=np.float64);w=w/w.sum()
            self.hidden_mean=(w[:,None]*h).sum(0).astype(np.float32)
            self.hidden_scale=max(float(np.sqrt((w[:,None]*(h-self.hidden_mean)**2).sum()/self.width)),1e-4)
        return self
    def transform(self,x):
        z=self.base.transform(x).astype(np.float32);parts=[]
        with torch.no_grad():
            for start in range(0,len(z),512):parts.append(self.net(torch.from_numpy((z[start:start+512]-self.mean)/self.sd)).numpy())
        h=np.concatenate(parts)
        return np.c_[z,(h-self.hidden_mean)/self.hidden_scale]

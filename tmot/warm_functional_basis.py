"""First-arrival supervised functions, then immutable exact-moment coordinates."""
import copy
import numpy as np
import torch
from torch import nn

class WarmFunctionalBasis:
    def __init__(self,base,task,mode='trained',seed=42,width=128,epochs=30,device='cuda'):
        self.base=base;self.task=task;self.mode=mode;self.seed=seed
        self.width=width;self.epochs=epochs;self.device=device;self.dimension=base.dimension+width
    def fit(self,x,y,calibration_x=None,sample_weight=None,input_stats=None,previous=None,distillation=0.,temperature=2.):
        if distillation<0 or temperature<=0:raise ValueError('Invalid distillation parameters')
        if hasattr(self,'net'):raise RuntimeError('Functional basis must remain fixed after first arrival')
        z=self.base.transform(x).astype(np.float32);y=np.asarray(y)
        cal=z if calibration_x is None else self.base.transform(calibration_x).astype(np.float32)
        self.mean=cal.mean(0);sd=cal.std(0);self.sd=np.maximum(sd,max(float(np.median(sd))*.01,1e-4))
        if input_stats is not None:self.mean,self.sd=(np.asarray(v,np.float32).copy() for v in input_stats)
        torch.manual_seed(self.seed)
        self.net=nn.Sequential(nn.Linear(self.base.dimension,self.width),nn.ReLU(),nn.Linear(self.width,self.width),nn.ReLU()).to(self.device)
        if self.task=='classification':
            labels=([] if previous is None else list(previous.classes));labels.extend(sorted(set(y.tolist())-set(labels)));self.classes=labels;ids=np.asarray([labels.index(c) for c in y]);outputs=len(labels);target=torch.as_tensor(ids,device=self.device)
            weights=1/np.bincount(ids)[ids];weights=weights/weights.sum()*len(y)
        else:
            if y.ndim==1:y=y[:,None]
            outputs=y.shape[1];target=torch.as_tensor(y,dtype=torch.float32,device=self.device);weights=np.ones(len(y))
        if sample_weight is not None:
            weights=np.asarray(sample_weight,dtype=np.float64)
            if weights.shape!=(len(y),) or np.any(weights<0) or not np.isfinite(weights).all() or weights.sum()<=0:raise ValueError('Invalid sample weights')
            weights=weights/weights.sum()*len(y)
        head=nn.Linear(self.width,outputs).to(self.device)
        self.warm_init_error=0.
        if previous is not None:
            self.net.load_state_dict(previous.net.state_dict())
            with torch.no_grad():
                head.weight[:previous.head.out_features].copy_(previous.head.weight)
                head.bias[:previous.head.out_features].copy_(previous.head.bias)
            self.warm_init_error=max(float((a.detach().cpu()-b.detach().cpu()).abs().max()) for a,b in zip(self.net.parameters(),previous.net.parameters()))
        for p in self.net.parameters():p.requires_grad_(True)
        features=torch.as_tensor((z-self.mean)/self.sd,device=self.device)
        weights=torch.as_tensor(weights,dtype=torch.float32,device=self.device)
        teacher=None
        if previous is not None and distillation:
            # Teacher uses its own fixed input calibration; never receives new labels.
            teacher_net=copy.deepcopy(previous.net).to(self.device).eval()
            teacher_head=copy.deepcopy(previous.head).to(self.device).eval()
            with torch.no_grad():
                teacher=teacher_head(teacher_net(torch.as_tensor((z-previous.mean)/previous.sd,device=self.device)))
            del teacher_net,teacher_head
        self.distillation=distillation;self.temperature=temperature;self.last_distillation_loss=0.
        self.optimizer_steps=0;self.last_loss=None
        if self.mode=='trained':
            opt=torch.optim.AdamW(list(self.net.parameters())+list(head.parameters()),lr=.001,weight_decay=1e-4)
            for _ in range(self.epochs):
                for batch in torch.randperm(len(y),device=self.device).split(256):
                    prediction=head(self.net(features[batch]))
                    error=nn.functional.cross_entropy(prediction,target[batch],reduction='none') if self.task=='classification' else (prediction-target[batch]).square().mean(1)
                    loss=(weights[batch]*error).mean()
                    if teacher is not None:
                        kd=distillation_error(prediction,teacher[batch],self.task,temperature)
                        kd=(weights[batch]*kd).mean()
                        loss=loss+distillation*kd;self.last_distillation_loss=float(kd.detach())
                    opt.zero_grad();loss.backward();opt.step()
                    self.optimizer_steps+=1;self.last_loss=float(loss.detach())
        elif self.mode!='random':raise ValueError(self.mode)
        self.head=head.to('cpu').eval()
        for p in self.head.parameters():p.requires_grad_(False)
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

    def predict_head(self,x):
        z=self.base.transform(x).astype(np.float32)
        with torch.no_grad():scores=self.head(self.net(torch.from_numpy((z-self.mean)/self.sd))).numpy()
        return np.asarray(self.classes)[scores.argmax(1)] if self.task=='classification' else scores


def distillation_error(student,teacher,task,temperature=2.):
    if task=='classification':
        # Old class ordering is preserved by WarmFunctionalBasis head expansion.
        old=student[:,:teacher.shape[1]]
        return nn.functional.kl_div(nn.functional.log_softmax(old/temperature,dim=1),nn.functional.softmax(teacher.detach()/temperature,dim=1),reduction='none').sum(1)*temperature**2
    return (student-teacher.detach()).square().mean(1)

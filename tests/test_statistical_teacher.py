import numpy as np
import torch
from tmot.statistical_teacher import distance_probabilities
from tmot.warm_functional_basis import distillation_error,external_distillation_error

def test_probabilities_scale_and_ties():
 d=np.array([[1.,3.,8.],[0.,0.,0.]])
 p=distance_probabilities(d);np.testing.assert_allclose(p,distance_probabilities(d*100));np.testing.assert_allclose(p.sum(1),1);np.testing.assert_allclose(p[1],1/3);assert p[0,0]>p[0,1]>p[0,2]

def test_full_target_detects_old_new_shift():
 a=torch.tensor([[1.,2.,3.]])
 b=torch.tensor([[-9.,-8.,3.]])
 old=a[:,:2]
 torch.testing.assert_close(distillation_error(a,old,'classification'),distillation_error(b,old,'classification'),atol=1e-6,rtol=0)
 p=torch.softmax(a,dim=1)
 assert external_distillation_error(b,p,'classification')>external_distillation_error(a,p,'classification')+1

def test_regression_distillation_numeric():
 a=torch.tensor([[1.,4.]],requires_grad=True);b=torch.tensor([[2.,2.]],requires_grad=True)
 loss=external_distillation_error(a,b,'regression');torch.testing.assert_close(loss,torch.tensor([2.5]));loss.sum().backward();assert b.grad is None

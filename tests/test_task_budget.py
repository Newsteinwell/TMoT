import numpy as np
from tmot.task_budget_replay import TaskBudgetER,TaskBudgetMomentER

def test_quota_and_missing_classes():
    for cls in (TaskBudgetER,TaskBudgetMomentER):
        m=cls(4,mode='er',epochs=1,device='cpu')
        for n,label in [(8,0),(11,2),(23,4)]:
            x=np.random.default_rng(label).normal(size=(n,4)).astype('float32');y=np.arange(n)%2+label
            m.observe(x,y)
            a=m.buffer_audit[-1]
            assert a['quota']==n//10 and len(set(a['selected_indices']))==n//10
            assert a['after']==sum(v['quota'] for v in m.buffer_audit)
            assert np.isfinite(m.predict(x)).all()
            assert abs(m.weight_summary['total']-1)<1e-12
        assert len(m.classes)==6

def test_regression_empty_and_nonempty():
    m=TaskBudgetER(4,task='regression',mode='er',epochs=1,device='cpu')
    for n in (5,21,30):
        x=np.random.default_rng(n).normal(size=(n,4)).astype('float32');m.observe(x,x[:,0])
        assert m.buffer_audit[-1]['quota']==n//10
        assert abs(m.weight_summary['total']-1)<1e-12
    assert len(m.buffer[0])==5

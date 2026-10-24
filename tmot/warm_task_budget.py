from .task_budget_replay import TaskBudgetER,TaskBudgetMomentER
from .warm_functional_basis import WarmFunctionalBasis
from .replay_functional import IdentityBasis

class WarmMixin:
    def fit_function(self,x,y,w):
        return WarmFunctionalBasis(IdentityBasis(self.dimension),self.task,seed=self.seed,epochs=self.epochs,device=self.device).fit(x,y,sample_weight=w,input_stats=self.input_stats,previous=getattr(self,'function',None))
    @property
    def persistent_bytes(self):
        return super().persistent_bytes+sum(p.numel()*p.element_size() for p in self.function.head.parameters())
class WarmER(WarmMixin,TaskBudgetER):pass
class WarmMomentER(WarmMixin,TaskBudgetMomentER):pass

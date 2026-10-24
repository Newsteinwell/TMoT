"""Nonlinear residual replay with a strict per-arrival 10% real buffer."""
from .task_budget_replay import TaskBudgetMixin
from .nonlinear_moments import NonlinearMomentLearner

class StrictResidualER(TaskBudgetMixin, NonlinearMomentLearner):
    def _historical(self, outputs):
        return self.sample_buffer() if self.stage else None

    def _store_buffer(self, x, y):
        self.store_buffer(x, y)

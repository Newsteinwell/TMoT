"""Optional persistence prior; first-arrival calibration, no retained samples."""
import numpy as np


class CausalReference:
    def __init__(self, mode):
        if mode not in ('none', 'append', 'residual'):
            raise ValueError(mode)
        self.mode = mode

    def fit(self, past):
        if hasattr(self, 'mean'):
            raise RuntimeError('First-arrival calibration is immutable')
        self.mean = np.asarray(past).mean(0)
        self.scale = np.maximum(np.asarray(past).std(0), 1e-6)
        return self

    def offset(self, past):
        return np.asarray(past) if self.mode == 'residual' else np.zeros_like(past)

    def features(self, x, past):
        if self.mode == 'none':
            return x
        return np.concatenate([x, (past-self.mean)/self.scale], axis=1).astype(np.float32)

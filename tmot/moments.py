"""Sufficient statistics in a FIXED basis; no historical individuals.

For z=[phi,1], target y: risk(F)=tr(F G F^T)-2<F,K^T>+tr(V),
where G=E[zz^T], K=E[zy^T], V=E[yy^T]. This identity is independent
of the input modality. Classification constructs y dynamically for seen classes.
"""
from dataclasses import dataclass
import numpy as np
import torch


def augment(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Expected finite N x d features')
    return np.column_stack([x, np.ones(len(x))])


@dataclass
class Cell:
    n: int
    total: np.ndarray
    gram: np.ndarray

    @classmethod
    def from_x(cls, x):
        z = augment(x)
        if len(z) == 0: raise ValueError('Empty cell')
        return cls(len(z), z.sum(0), z.T @ z)

    def merge(self, other):
        self.n += other.n
        self.total += other.total
        self.gram += other.gram


class ClassificationMoments:
    """Per-class cumulative raw moments; missing classes have no fabricated cell.

    Class-balanced global head, no task/domain ID at inference. Domains are
    pooled within class by sample count in this first implementation.
    """
    def __init__(self, dimension):
        self.dimension = dimension
        self.cells = {}

    def update(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        if x.ndim != 2 or x.shape[1] != self.dimension or y.shape != (len(x),):
            raise ValueError('Shape mismatch')
        for c in np.unique(y):
            new = Cell.from_x(x[y == c])
            if int(c) in self.cells: self.cells[int(c)].merge(new)
            else: self.cells[int(c)] = new

    def risk_moments(self, classes):
        classes = list(classes)
        if not set(self.cells).issubset(classes): raise ValueError('Old classes missing from output')
        if not self.cells: raise ValueError('No observations')
        g = np.zeros((self.dimension+1, self.dimension+1))
        k = np.zeros((self.dimension+1, len(classes)))
        for c, cell in self.cells.items():
            g += cell.gram/cell.n
            k[:, classes.index(c)] += cell.total/cell.n
        return g/len(self.cells), k/len(self.cells), 1.0

    @property
    def nbytes(self):
        return sum(c.total.nbytes+c.gram.nbytes+8 for c in self.cells.values())


class RegressionMoments:
    """Sample-weighted input/target moments, supports scalar or vector regression."""
    def __init__(self, dimension, outputs):
        self.dimension, self.outputs = dimension, outputs
        self.n = 0
        self.gram = np.zeros((dimension+1, dimension+1))
        self.cross = np.zeros((dimension+1, outputs))
        self.target_gram = np.zeros((outputs, outputs))

    def update(self, x, y):
        z = augment(x); y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1: y = y[:, None]
        if z.shape[1] != self.dimension+1 or y.shape != (len(z), self.outputs):
            raise ValueError('Input/target shape mismatch')
        if not np.isfinite(y).all(): raise ValueError('Nonfinite targets')
        self.n += len(z); self.gram += z.T @ z
        self.cross += z.T @ y; self.target_gram += y.T @ y

    def risk_moments(self, classes=None):
        if self.n == 0: raise ValueError('No observations')
        return self.gram/self.n, self.cross/self.n, np.trace(self.target_gram)/self.n

    @property
    def nbytes(self):
        return self.gram.nbytes+self.cross.nbytes+self.target_gram.nbytes+8


def affine_matrix(weight, bias):
    """Homogeneous matrix H such that [h,1] = [phi,1] H^T."""
    a = np.asarray(weight); b = np.asarray(bias)
    out = np.zeros((a.shape[0]+1, a.shape[1]+1))
    out[:-1, :-1] = a; out[:-1, -1] = b; out[-1, -1] = 1
    return out


def transport_risk(g, k, h):
    return h @ g @ h.T, h @ k


def quadratic_risk(f, g, k, target_sq):
    """Unclamped differentiable risk; clamp would hide numerical defects."""
    return (f * (f @ g)).sum() - 2 * (f * k.T).sum() + target_sq


def tensor_moments(memory, classes, device):
    g,k,v=memory.risk_moments(classes)
    return tuple(torch.as_tensor(a,dtype=torch.float64,device=device) for a in (g,k,v))


def ridge_readout(g, k, ridge=0.01):
    penalty = np.eye(len(g))*ridge
    penalty[-1,-1] = 0
    return np.linalg.solve((g+g.T)/2+penalty, k)

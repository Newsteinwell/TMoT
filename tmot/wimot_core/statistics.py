"""Class-conditional moments of the frozen basis, their storage formats and the exact affine transport to h."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TaskStatistics:
    counts: np.ndarray       # (C,)
    means: np.ndarray        # (C, d)
    covariances: np.ndarray  # (C, d, d)


def matrix(X, d=None):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or not X.shape[1] or not np.isfinite(X).all(): raise ValueError('Expected finite feature matrix')
    if d is not None and X.shape[1] != d: raise ValueError('Feature dimension changed')
    return X


def moments(X, y, classes):
    X = matrix(X); y = np.asarray(y)
    if y.shape != (len(X),) or y.dtype.kind not in 'iu': raise ValueError('Integer labels required')
    if not np.array_equal(np.unique(y), np.arange(classes)): raise ValueError('Each domain must contain all classes')
    rows = [X[y == c] for c in range(classes)]
    if min(map(len, rows)) < 2: raise ValueError('Need at least two windows per class')
    return TaskStatistics(np.array([len(r) for r in rows], dtype=float),
                          np.stack([r.mean(0) for r in rows]),
                          np.stack([np.atleast_2d(np.cov(r, rowvar=False)) for r in rows]))


def stored(stats, storage=None, rank=None):
    """Covariances as reconstructed from the stored representation: dense (None), upper triangle, or rank-r + isotropic."""
    if storage is None: return stats
    cov = stats.covariances
    if storage == 'triu':
        iu = np.triu_indices(cov.shape[-1])
        packed = cov[:, iu[0], iu[1]]; out = np.zeros_like(cov)
        out[:, iu[0], iu[1]] = packed; out[:, iu[1], iu[0]] = packed
    elif storage == 'rank':
        out = np.empty_like(cov)
        for c in range(len(cov)):
            w, v = np.linalg.eigh((cov[c]+cov[c].T)/2); w, v = w[::-1], v[:, ::-1]
            rest = w[rank:].mean(); u = v[:, :rank]
            out[c] = (u*(w[:rank]-rest))@u.T+rest*np.eye(len(w))
    else:
        raise ValueError(f'Unknown storage {storage}')
    return TaskStatistics(stats.counts, stats.means, out)


def pooled(stats):
    counts = np.stack([s.counts for s in stats])
    means = np.stack([s.means for s in stats])
    covs = np.stack([s.covariances for s in stats])
    n = counts.sum(0)
    mu = np.einsum('tc,tcd->cd', counts, means)/n[:, None]
    delta = means-mu[None]
    scatter = np.einsum('tc,tcij->cij', counts-1, covs)
    scatter += np.einsum('tc,tci,tcj->cij', counts, delta, delta)
    # The accumulation order is kept fixed: finite-difference OSCL optimisation amplifies last-bit changes.
    within = np.einsum('tc,tcij->ij', counts-1, covs)
    within += np.einsum('tc,tci,tcj->ij', counts, delta, delta)
    within /= n.sum()-len(n)
    within = (within+within.T)/2
    center = np.einsum('c,cd->d', n, mu)/n.sum()
    merged = TaskStatistics(n, mu, scatter/(n-1)[:, None, None])
    return merged, within, center


def root(cov, shrinkage=.1):
    """Shrunk inverse square root, inverse and log-determinant of a covariance."""
    if not 0 < shrinkage <= 1: raise ValueError('Shrinkage must be in (0,1]')
    scale = max(float(np.trace(cov)/len(cov)), 1e-12)
    cov = (1-shrinkage)*cov+shrinkage*scale*np.eye(len(cov))
    vals, vecs = np.linalg.eigh((cov+cov.T)/2)
    vals = np.maximum(vals, 1e-9*scale)
    return (vecs/np.sqrt(vals))@vecs.T, (vecs/vals)@vecs.T, np.log(vals).sum()


def transport(stats, model):
    """Exact moments of h = A phi + b from stored phi moments."""
    a = model.adapter.weight.detach().cpu().numpy().astype(np.float64)
    b = model.adapter.bias.detach().cpu().numpy().astype(np.float64)
    if a.shape != (128, 512): raise ValueError('512 -> 128 adapter required')
    return [TaskStatistics(s.counts.copy(), s.means@a.T+b, np.array([a@c@a.T for c in s.covariances])) for s in stats]


def save_stats(path, stats):
    np.savez_compressed(path, counts=np.stack([s.counts for s in stats]), means=np.stack([s.means for s in stats]),
                        covariances=np.stack([s.covariances for s in stats]))


def load_stats(path):
    with np.load(path) as f:
        return [TaskStatistics(n.copy(), m.copy(), c.copy()) for n, m, c in zip(f['counts'], f['means'], f['covariances'])]


def statistic_bytes(stats):
    return sum(s.counts.nbytes+s.means.nbytes+s.covariances.nbytes for s in stats)

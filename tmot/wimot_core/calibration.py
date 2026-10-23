"""Per-stage choice of the readout density case by two-fold, head-only cross-fitting on TRAIN data available at that stage."""
import numpy as np
from scipy.special import softmax

from .readout import StudentDecoder, case_key, cases, head_from
from .statistics import TaskStatistics, transport


def folds(labels, tasks, seed=42):
    y = np.asarray(labels); t = np.asarray(tasks); out = np.full(len(y), -1, dtype=np.int64)
    for task in np.unique(t):
        for c in np.unique(y):
            ids = np.flatnonzero((t == task) & (y == c))
            if not len(ids): continue
            if len(ids) == 1:
                out[ids] = (int(task)+int(c)) % 2; continue
            order = np.random.default_rng(seed+1009*int(task)+9176*int(c)).permutation(ids)
            out[order] = np.arange(len(ids)) % 2
    assert np.all(out >= 0)
    return out


def weights(y, t):
    """Equal weight per (domain, class) cell."""
    w = np.zeros(len(y), float); tasks = np.unique(t)
    for task in tasks:
        classes = np.unique(y[t == task])
        for c in classes:
            ids = (t == task) & (y == c)
            w[ids] = 1/(len(tasks)*len(classes)*ids.sum())
    return w


def subtract(stats, x, y, t):
    """Remove held-out rows from the stored moments by centred scatter arithmetic."""
    out = []
    for task, s in enumerate(stats):
        counts = s.counts.copy(); means = s.means.copy(); covs = s.covariances.copy()
        for c in range(len(counts)):
            rows = x[(t == task) & (y == c)]; n = float(counts[c]); m = len(rows); remaining = n-m
            if not m: continue
            if remaining < 2: raise ValueError('Insufficient remaining class count')
            mu = (n*means[c]-rows.sum(0))/remaining
            d = rows-means[c]; shift = mu-means[c]
            scatter = (n-1)*covs[c]-d.T@d-remaining*np.outer(shift, shift)
            counts[c] = remaining; means[c] = mu; covs[c] = (scatter+scatter.T)/(2*(remaining-1))
        out.append(TaskStatistics(counts, means, covs))
    return out


def normalized(p):
    """Probability renormalisation used for every readout output (log, then softmax)."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, rtol=1e-7, atol=1e-7):
        raise ValueError('Normalised finite probabilities required')
    return softmax(1.*np.log(np.maximum(p, 1e-300))/1., axis=1)


def select(probabilities, y, t):
    w = weights(y, t); best = None; diagnostics = []
    for case, p in probabilities.items():
        q = normalized(p)
        value = dict(case=case, NLL=float(-w@np.log(np.maximum(q[np.arange(len(y)), y], 1e-300))),
                     accuracy=float(w@(q.argmax(1) == y)))
        diagnostics.append(value)
        if best is None or value['NLL'] < best['NLL']: best = value
    return best, diagnostics


def head_calibration(model, stats, x, y, t, seed=42, h_stats=None):
    """Choose (df, gate) on the rows (x, y, t) available at this stage; x are frozen-basis phi rows.

    ``stats`` are phi-coordinate moments, transported through the current adapter. ``h_stats`` (the no-transport ablation
    only) are moments already in h coordinates, each in the coordinates of the adapter of its own domain; they are used as given.
    """
    hs = transport(stats, model) if h_stats is None else h_stats
    h = x.astype(float)@model.adapter.weight.detach().cpu().numpy().T+model.adapter.bias.detach().cpu().numpy()
    assign = folds(y, t, seed=seed)
    classes = len(stats[0].counts)
    probabilities = {case_key(c): np.zeros((len(y), classes)) for c in cases()}
    for fold in (0, 1):
        ids = np.flatnonzero(assign == fold)
        base = head_from(subtract(hs, h[ids], y[ids], t[ids]), classes)
        for case in cases(): probabilities[case_key(case)][ids] = StudentDecoder(base, *case).predict_proba(h[ids])
    selected, diagnostics = select(probabilities, y, t)
    return dict(selected=selected, diagnostics=diagnostics, rows=len(y), historical_query_rows=int(np.sum(t < t.max())))

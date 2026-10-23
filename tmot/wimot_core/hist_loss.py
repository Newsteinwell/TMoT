"""Historical moment-consistency loss L_hist.

For the centred affine score map F = centre_rows([W A, W b + d]), each stored domain contributes
the closed-form squared error tr(F S F^T) - 2 <F, K> + 1 against unit-norm centred one-hot targets,
where S is the augmented second moment and K the target cross moment of that domain.
"""
import numpy as np
import torch


def second_moment(stats):
    """Equal-weight class raw second moment of [phi, 1]."""
    rows = []
    for s in stats:
        for n, mu, cov in zip(s.counts, s.means, s.covariances):
            block = np.empty((len(mu)+1, len(mu)+1), dtype=np.float64)
            block[:-1, :-1] = (n-1)/n*cov+np.outer(mu, mu)
            block[:-1, -1] = mu; block[-1, :-1] = mu; block[-1, -1] = 1
            rows.append(block)
    return np.mean(rows, axis=0)


def centered_affine(model):
    a, b = model.adapter.weight, model.adapter.bias
    c, d = model.classifier.weight, model.classifier.bias
    augmented = torch.cat([c@a, (c@b+d)[:, None]], dim=1)
    return augmented-augmented.mean(dim=0, keepdim=True)


def class_moments(stats):
    counts = np.asarray(stats.counts); means = np.asarray(stats.means, dtype=np.float64)
    classes = len(counts)
    if classes < 2 or np.any(counts < 2) or means.ndim != 2 or len(means) != classes:
        raise ValueError('At least two classes and two samples per class required')
    if not np.isfinite(means).all() or not np.isfinite(stats.covariances).all(): raise ValueError('Finite moments required')
    targets = (np.eye(classes)-np.ones((classes, classes))/classes)/np.sqrt(1-1/classes)
    augmented = np.column_stack([means, np.ones(classes)])
    return second_moment([stats]), targets.T@augmented/classes


def first_scale(model, stats):
    f = centered_affine(model).detach()
    second = torch.as_tensor(class_moments(stats)[0], dtype=f.dtype, device=f.device)
    return (f*(f@second)).sum().clamp_min(1e-16).sqrt().detach()


def constraint(model, stats):
    f = centered_affine(model)
    if f.shape[0] != len(stats.counts): raise ValueError('Classifier/class moment mismatch')
    second, cross = class_moments(stats)
    return tuple(torch.as_tensor(v, dtype=f.dtype, device=f.device).detach().clone() for v in (second, cross))


def loss(model, constraints, scale):
    f = centered_affine(model)
    if not constraints: return f.sum()*0
    scale = torch.as_tensor(scale, dtype=f.dtype, device=f.device).detach()
    if scale.numel() != 1 or not torch.isfinite(scale) or scale <= 0: raise ValueError('Positive first-domain scale required')
    f = f/scale; values = []
    for second, cross in constraints:
        value = (f*(f@second)).sum()-2*(f*cross).sum()+1
        values.append(value.clamp_min(0))
    return torch.stack(values).mean()


def state_bytes(constraints, scale):
    s = torch.as_tensor(scale)
    return sum(t.numel()*t.element_size() for row in constraints for t in row)+s.numel()*s.element_size()

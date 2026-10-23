"""OSCL readout fitted from stored class moments only (no raw historical samples).

Fitting chain for one stage (``head_from``):
  1. ``DensityBank``: one shared whitening of the pooled moments; per-domain Gaussian class experts with a
     dimension-scaled covariance prior; it provides the domain gate.
  2. ``PreconditionedOSCL``: per domain, a within-domain preconditioner and pairwise OSCL directions obtained by
     minimising the log Gaussian-intersection risk of every class pair (L-BFGS-B, finite differences).
  3. ``SpectralDecoder``: the OSCL direction bank of each domain is reduced to its spectral subspace and the
     class densities are re-estimated there.
  4. ``MeanPriorDecoder``: the subspace is blended with the whitened class-mean directions.
``StudentDecoder`` replaces Gaussian class densities (and optionally the gate) by Student-t densities with the
same covariance; its case (df, gate) is chosen per stage by two-fold TRAIN NLL (see calibration.py).
"""
import copy
import hashlib
from itertools import combinations

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import gammaln, log_ndtr, logsumexp, ndtr

from .protocol import PROTOCOL
from .statistics import TaskStatistics, matrix, pooled, root

R = PROTOCOL['method']['readout']


def classification_error(m1, v1, m2, v2):
    """Equal-prior Gaussian-intersection error and threshold of two 1-D Gaussians (root between the means)."""
    s1, s2 = np.sqrt(max(v1, 1e-30)), np.sqrt(max(v2, 1e-30))
    if m1 > m2: m1, m2, s1, s2 = m2, m1, s2, s1
    threshold = 0.5*(m1+m2)
    if abs(s1-s2) >= 1e-12:
        a = s1**2-s2**2
        b = 2*(m1*s2**2-m2*s1**2)
        c = (m2**2*s1**2-m1**2*s2**2
             - 2*s1**2*s2**2*np.log(s1/s2+1e-30))
        disc = b**2-4*a*c
        if disc >= 0:
            x1 = (-b+np.sqrt(disc))/(2*a)
            x2 = (-b-np.sqrt(disc))/(2*a)
            if m1 <= x1 <= m2: threshold = x1
            elif m1 <= x2 <= m2: threshold = x2
    error = 0.5*ndtr((m1-threshold)/s1)+0.5*ndtr((threshold-m2)/s2)
    return float(error), float(threshold)


class PairOSCL:
    """Projected moments of one class pair along w = sum_k c_k l_k mu_k; slots are domain-major, then pair order."""
    def __init__(self, task_stats, pair, target):
        self.target = target
        self.pair = tuple(pair)
        self.means = np.concatenate([s.means[list(pair)] for s in task_stats])
        self.counts = np.concatenate([s.counts[list(pair)] for s in task_stats])
        covariances = np.concatenate([s.covariances[list(pair)] for s in task_stats])
        self.num_tasks = len(task_stats)
        self.c = self.counts/self.counts.sum()
        weighted_means = self.c[:, None]*self.means
        self.mean_matrix = weighted_means@self.means.T
        traces = np.einsum('kab,jab->jk', covariances, covariances, optimize=True)/self.counts.sum()
        self.variance_matrices = np.array([weighted_means@cov@weighted_means.T+np.diag(self.c*trace)
                                           for cov, trace in zip(covariances, traces)])
        self.set_labels(np.tile([-1., 1.], self.num_tasks))

    def moments(self, labels=None):
        labels = self.labels if labels is None else np.asarray(labels, dtype=float)
        means = labels@self.mean_matrix
        variances = np.einsum('k,jkl,l->j', labels, self.variance_matrices, labels, optimize=True)
        return means.reshape(-1, 2), variances.reshape(-1, 2)

    def risk_rows(self, labels):
        means, variances = self.moments(labels)
        return means[self.target:self.target+1], variances[self.target:self.target+1]

    def objective(self, labels):
        means, variances = self.risk_rows(labels)
        return float(np.mean([classification_error(m[0], v[0], m[1], v[1])[0] for m, v in zip(means, variances)]))

    def log_objective(self, labels):
        means, variances = self.risk_rows(labels)
        values = []
        for m, v in zip(means, variances):
            threshold = classification_error(m[0], v[0], m[1], v[1])[1]
            if m[0] > m[1]: m, v = m[::-1], v[::-1]
            sd = np.sqrt(np.maximum(v, 1e-30))
            values.append(logsumexp([log_ndtr((m[0]-threshold)/sd[0]), log_ndtr((threshold-m[1])/sd[1])])-np.log(2.))
        return float(logsumexp(values)-np.log(len(values)))

    def set_labels(self, labels):
        labels = np.asarray(labels, dtype=float)
        if labels.shape != self.counts.shape or not np.isfinite(labels).all(): raise ValueError('Invalid combining labels')
        self.labels = labels.copy()
        self.weight = (self.c*labels)@self.means
        means, variances = self.moments()
        self.thresholds = np.array([classification_error(m[0], v[0], m[1], v[1])[1] for m, v in zip(means, variances)])


class TargetRiskPair(PairOSCL):
    """Combining labels optimised for the log risk of the target domain, accepted only if not worse than fixed labels."""
    damping = 1.

    def __init__(self, stats, pair, target, maxiter=200):
        super().__init__(stats, pair, target)
        initial = self.labels.copy()
        fixed_log = self.log_objective(initial)
        candidate, value, info = fmin_l_bfgs_b(self.log_objective, initial, bounds=[(-10., 10.)]*len(initial),
                                               approx_grad=True, maxiter=maxiter, pgtol=1e-6)
        accepted = bool(np.isfinite(candidate).all() and np.isfinite(value) and value <= fixed_log+1e-10)
        candidate = candidate if accepted else initial
        if candidate@initial < 0: candidate = -candidate
        length = np.linalg.norm(candidate)
        candidate = candidate*np.linalg.norm(initial)/length if length > 1e-12 else initial
        labels = (1-self.damping)*initial+self.damping*candidate
        accepted = bool(accepted and np.isfinite(self.log_objective(labels)) and self.log_objective(labels) <= fixed_log+1e-10)
        if not accepted: labels = initial
        self.set_labels(labels)
        self.status = dict(accepted=accepted, iterations=int(info['nit']), log_objective=self.log_objective(labels),
                           fixed_log_objective=fixed_log)


def prior_coefficients(counts, dimension, kappa):
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or not np.isfinite(counts).all() or np.any(counts < 2) or np.any(counts != np.floor(counts)):
        raise ValueError('Observed integer class counts >= 2 required')
    pseudo = float(kappa)*dimension
    return pseudo/(counts-1+pseudo)


def covariance_prior(counts, covariances, kappa):
    cov = np.asarray(covariances, dtype=np.float64)
    d = cov.shape[1]; beta = prior_coefficients(counts, d, kappa)
    return (1-beta[:, None, None])*cov+beta[:, None, None]*np.eye(d)[None]


class DensityBank:
    """Globally whitened per-domain Gaussian class experts; the gate is p(domain | h) under equal priors."""
    def __init__(self, classes, kappa=R['kappa'], rho=R['white_rho'], maxiter=R['maxiter']):
        self.classes, self.kappa, self.rho, self.maxiter = classes, float(kappa), rho, maxiter

    def fit_statistics(self, stats):
        self.raw_stats = list(stats); self.d = stats[0].means.shape[1]; self.tasks = len(stats)
        if any(np.any(s.counts < 2) or s.means.shape != (self.classes, self.d)
               or s.covariances.shape != (self.classes, self.d, self.d) for s in stats):
            raise ValueError('Invalid task moments')
        self.pairs = tuple(combinations(range(self.classes), 2))
        self.experts = []
        _, cov, center = pooled(stats)
        transform = root(cov, self.rho)[0]
        sign, logjac = np.linalg.slogdet(transform)
        if sign <= 0: raise ValueError('Whitening determinant must be positive')
        transformed = []
        for s in stats:
            cs = np.einsum('ij,cjk,kl->cil', transform.T, s.covariances, transform, optimize=True)
            transformed.append(TaskStatistics(s.counts, (s.means-center)@transform, covariance_prior(s.counts, cs, self.kappa)))
        for target in range(len(stats)):
            mu, cs = transformed[target].means, transformed[target].covariances
            density = [root(c, 1e-6)[1:] for c in cs]
            self.experts.append(dict(center=center, transform=transform, logjac=float(logjac), means=mu, covariances=cs,
                                     precisions=np.array([r[0] for r in density]), logdets=np.array([r[1] for r in density])))
        return self

    def component_logpdf(self, x):
        x = matrix(x, self.d); terms = []
        for e in self.experts:
            white = (x-e['center'])@e['transform']
            delta = white[:, None, :]-e['means'][None]
            quadratic = np.einsum('nci,cij,ncj->nc', delta, e['precisions'], delta, optimize=True)
            terms.append(-.5*(quadratic+e['logdets'][None]+self.d*np.log(2*np.pi))+e['logjac'])
        return np.stack(terms, axis=1)

    def gate_probabilities(self, x):
        log_gate = logsumexp(self.component_logpdf(x), axis=2)-np.log(self.classes)
        return np.exp(log_gate-logsumexp(log_gate, axis=1, keepdims=True))


class PreconditionedOSCL:
    """Per-domain within-domain preconditioning of the DensityBank experts, then target-risk pairwise OSCL."""
    def __init__(self, provider, rho=R['precondition_rho']):
        self.__dict__ = copy.copy(provider.__dict__)
        self.provider = provider; self.cache = {}; self.experts = []
        for target, original in enumerate(provider.experts):
            source = provider.raw_stats[target]
            within = sum((n-1)*c for n, c in zip(source.counts, original['covariances']))/sum(source.counts-1)
            preconditioner = root(within, rho)[0]
            moments = [TaskStatistics(s.counts, e['means']@preconditioner,
                                      np.array([preconditioner.T@c@preconditioner for c in e['covariances']]))
                       for s, e in zip(provider.raw_stats, provider.experts)]
            s = moments[target]; e = copy.copy(original)
            e.update(transform=original['transform']@preconditioner, means=s.means, covariances=s.covariances,
                     logjac=float(original['logjac']+np.linalg.slogdet(preconditioner)[1]))
            density = [root(c, 1e-6)[1:] for c in s.covariances]
            e.update(precisions=np.array([d[0] for d in density]), logdets=np.array([d[1] for d in density]))
            models = [TargetRiskPair(moments, p, target, self.maxiter) for p in self.pairs]
            e.update(weights=np.stack([m.weight for m in models]))
            self.experts.append(e)

    def gate_probabilities(self, x):
        key = hashlib.sha256(np.ascontiguousarray(x, dtype=np.float64).tobytes()).hexdigest()
        if key not in self.cache: self.cache[key] = self.provider.gate_probabilities(x)
        return self.cache[key]


def projection(rows):
    rows = np.asarray(rows, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1); nonzero = norms > 1e-12
    if not nonzero.any(): raise ValueError('Degenerate direction bank')
    unit = rows[nonzero]/norms[nonzero, None]
    _, singular, vt = np.linalg.svd(unit, full_matrices=False)
    q = int(np.sum(singular > 1e-8*singular[0]))
    return vt[:q].copy()*(singular[:q, None]/np.sqrt(len(unit)))


def regularize(counts, covariances, alpha, gamma):
    counts = np.asarray(counts, float); covariances = np.asarray(covariances, float)
    pooled_cov = sum((n-1)*s for n, s in zip(counts, covariances))/sum(counts-1)
    mixed = (1-alpha)*covariances+alpha*pooled_cov[None]
    q = mixed.shape[-1]; scale = np.maximum(np.trace(mixed, axis1=-2, axis2=-1)/q, 1e-12)
    covariance = (1-gamma)*mixed+gamma*scale[:, None, None]*np.eye(q)[None]
    covariance = (covariance+covariance.swapaxes(-1, -2))/2
    vals, vecs = np.linalg.eigh(covariance); vals = np.maximum(vals, 1e-9*scale[:, None])
    covariance = np.einsum('cqi,ci,cri->cqr', vecs, vals, vecs)
    precision = np.einsum('cqi,ci,cri->cqr', vecs, 1/vals, vecs)
    return covariance, precision, np.log(vals).sum(-1)


class SpectralDecoder:
    """Class densities of each domain inside the spectral subspace of its OSCL direction bank."""
    def __init__(self, base, alpha=R['alpha'], gamma=R['gamma']):
        self.base = base; self.classes = base.classes; self.d = base.d
        self.alpha, self.gamma = alpha, gamma; self.experts = []
        for task, e in enumerate(base.experts):
            if np.any(np.linalg.norm(e['weights'], axis=1) <= 1e-12): raise ValueError('Degenerate OSCL pair direction')
            directions = projection(e['weights'])
            raw = np.array([directions@s@directions.T for s in e['covariances']])
            cov, precision, logdets = regularize(base.raw_stats[task].counts, raw, alpha, gamma)
            self.experts.append(dict(directions=directions, means=e['means']@directions.T, covariances=cov,
                                     precisions=precision, logdets=logdets, center=e['center'],
                                     transform=e['transform'], logjac=e['logjac']))

    def conditional_logpdf(self, x):
        x = matrix(x, self.d); pieces = []
        for e in self.experts:
            z = ((x-e['center'])@e['transform'])@e['directions'].T
            delta = z[:, None, :]-e['means'][None]
            quadratic = np.einsum('nci,cij,ncj->nc', delta, e['precisions'], delta, optimize=True)
            pieces.append(-.5*(quadratic+e['logdets'][None]+z.shape[1]*np.log(2*np.pi)))
        return np.stack(pieces, axis=1)

    def predict_proba(self, x):
        terms = self.conditional_logpdf(x)
        conditional = np.exp(terms-logsumexp(terms, axis=2, keepdims=True))
        return np.einsum('nt,ntc->nc', self.base.gate_probabilities(x), conditional)


def mean_rows(base, expert):
    rows = np.concatenate([(s.means-expert['center'])@expert['transform'] for s in base.raw_stats])
    norms = np.linalg.norm(rows, axis=1); keep = norms > 1e-12
    if not keep.any(): raise ValueError('Nonzero class means required')
    return rows[keep]/norms[keep, None]


def blended_directions(directions, unit_means, strength):
    d = np.asarray(directions, float); u = np.asarray(unit_means, float)
    energy = float(np.square(d).sum())
    if energy <= 0: raise ValueError('Nonzero direction bank required')
    rows = np.concatenate([np.sqrt((1-strength)/energy)*d, np.sqrt(strength/len(u))*u])
    _, singular, vt = np.linalg.svd(rows, full_matrices=False)
    q = int(np.sum(singular > 1e-8*singular[0]))
    return singular[:q, None]*vt[:q]


class MeanPriorDecoder(SpectralDecoder):
    """SpectralDecoder whose subspace is blended with whitened class-mean directions (strength 0.5)."""
    def __init__(self, reference, strength=R['mean_prior_strength']):
        self.__dict__.update(reference.__dict__); self.experts = []
        for task, old in enumerate(reference.experts):
            base = reference.base.experts[task]
            d = blended_directions(old['directions'], mean_rows(reference.base, base), strength)
            raw = np.array([d@s@d.T for s in base['covariances']])
            cov, precision, logdets = regularize(reference.base.raw_stats[task].counts, raw, self.alpha, self.gamma)
            self.experts.append(dict(directions=d, means=base['means']@d.T, covariances=cov, precisions=precision,
                                     logdets=logdets, center=old['center'], transform=old['transform'], logjac=old['logjac']))


def head_from(stats, classes):
    """Fit the OSCL readout of one stage from h-space moments of every stored domain."""
    provider = DensityBank(classes).fit_statistics(stats)
    return MeanPriorDecoder(SpectralDecoder(PreconditionedOSCL(provider)))


def density(x, expert, df, directions=None):
    z = (x-expert['center'])@expert['transform']
    if directions is not None: z = z@directions.T
    delta = z[:, None, :]-expert['means'][None]
    quadratic = np.einsum('nci,cij,ncj->nc', delta, expert['precisions'], delta, optimize=True)
    q = z.shape[1]
    if df is None: return -.5*(quadratic+expert['logdets'][None]+q*np.log(2*np.pi))
    scale = (df-2)/df
    return (gammaln((df+q)/2)-gammaln(df/2)-.5*(q*np.log(df*np.pi)+expert['logdets'][None]+q*np.log(scale))
            - .5*(df+q)*np.log1p(np.maximum(quadratic, 0)/(df*scale)))


def cases():
    return [(None, 'gaussian')]+[(nu, g) for nu in R['degrees_of_freedom'] for g in R['gate_modes']]


def case_key(c):
    return f"nu{'inf' if c[0] is None else c[0]}/gate{c[1]}"


def case_from_key(key):
    return next(c for c in cases() if case_key(c) == key)


class StudentDecoder:
    """Class densities (and optionally the domain gate) with Student-t tails of equal covariance."""
    def __init__(self, reference, df, gate):
        if gate not in ('gaussian', 'student') or (df is None and gate != 'gaussian'): raise ValueError('Unknown density case')
        if df is not None and (not np.isfinite(df) or df <= 2): raise ValueError('df > 2 required')
        self.reference, self.df, self.gate = reference, df, gate
        self.d, self.classes = reference.d, reference.classes

    def gate_probabilities(self, x):
        x = matrix(x, self.d)
        if self.gate == 'gaussian': return self.reference.base.gate_probabilities(x)
        provider = self.reference.base.provider
        terms = np.stack([density(x, e, self.df)+e['logjac'] for e in provider.experts], axis=1)
        log_gate = logsumexp(terms, axis=2)-np.log(self.classes)
        return np.exp(log_gate-logsumexp(log_gate, axis=1, keepdims=True))

    def predict_proba(self, x):
        if self.df is None: return self.reference.predict_proba(x)
        x = matrix(x, self.d)
        terms = np.stack([density(x, e, self.df, e['directions']) for e in self.reference.experts], axis=1)
        conditional = np.exp(terms-logsumexp(terms, axis=2, keepdims=True))
        return np.einsum('nt,ntc->nc', self.gate_probabilities(x), conditional)

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)   # results (JSON) and figures are read/written in this folder
import sys, time, numpy as np
from phlm import pwm_dense, pwm_spectral, lmoments_from_pwm, sample_lmoments, cf1
from scipy.optimize import minimize
from scipy.linalg import expm
from math import factorial
import pyreadr

def load(name):
    d = _os.path.join(_ROOT, "data") + _os.sep
    if name == "auto":
        x = pyreadr.read_r(d + "AutoClaims.rda")["AutoClaims"]["PAID"].values
    elif name == "danish":
        x = pyreadr.read_r(d + "x_danishuni.rda")["danishuni"]["Loss"].values
    else:   # "danishx": excesses over the 1 MDKK reporting threshold
        x = pyreadr.read_r(d + "x_danishuni.rda")["danishuni"]["Loss"].values - 1.0
        x = np.maximum(x, 0.0)
    return np.asarray(x, float)

# ---------------- PH utilities ----------------
def ph_lm(alpha, S, R):
    """L-moments with automatic route choice (Algorithm 1)."""
    m = S.shape[0]
    eta, V = np.linalg.eig(S)
    if np.finfo(float).eps * np.linalg.cond(V) ** R < 1e-8:
        return lmoments_from_pwm(pwm_spectral(alpha, S, R - 1))
    return lmoments_from_pwm(pwm_dense(alpha, S, R - 1))

def ph_moments(alpha, S, R):
    U = np.linalg.inv(-S); v = np.ones(S.shape[0]); out = []
    for k in range(1, R + 1):
        v = U @ v; out.append(factorial(k) * alpha @ v)
    return np.array(out)

def ph_cdf(alpha, S, x):
    x = np.atleast_1d(x); eta, V = np.linalg.eig(S); Vi = np.linalg.inv(V)
    a = alpha @ V; b = Vi @ np.ones(len(alpha))
    return 1 - np.real((np.exp(np.outer(x, eta)) * (a * b)).sum(1))

def ph_logpdf(alpha, S, x):
    s = -S @ np.ones(len(alpha)); eta, V = np.linalg.eig(S); Vi = np.linalg.inv(V)
    a = alpha @ V; c = Vi @ s
    return np.log(np.maximum(np.real((np.exp(np.outer(x, eta)) * (a * c)).sum(1)), 1e-300))

# ---------------- CF-1 parameterisation ----------------
def unpack_cf1(t, m):
    inc = np.exp(t[:m]); rates = np.cumsum(inc)            # ascending, distinct
    w = np.exp(np.r_[t[m:], 0.0]); alpha = w / w.sum()       # softmax, last logit fixed 0
    return cf1(rates, alpha)

def fit_cf1(target, m, kind="L", starts=20, seed=0, w=None, init=None):
    """kind 'L': match L-moments; kind 'C': match raw moments. target on the scaled data."""
    R = len(target); w = np.ones(R) if w is None else w
    fun = (lambda a, S: ph_lm(a, S, R)) if kind == "L" else (lambda a, S: ph_moments(a, S, R))
    def obj(t):
        try:
            a, S = unpack_cf1(t, m); v = fun(a, S)
            return float(np.sum(w * (v / target - 1) ** 2)) if np.all(np.isfinite(v)) else 1e6
        except Exception:
            return 1e6
    rng = np.random.default_rng(seed); best = None; t0 = time.time(); results = []
    inits = [] if init is None else list(init)
    for k in range(starts):
        inits.append(np.r_[rng.normal(np.log(1.0 / m), 1.0, m), rng.normal(0, 1, m - 1)])
    for t in inits:
        r = minimize(obj, t, method="L-BFGS-B", bounds=[(-12, 6)] * m + [(-25, 25)] * (m - 1),
                     options=dict(ftol=1e-14, gtol=1e-10, maxiter=2000))
        results.append(r.fun)
        if best is None or r.fun < best.fun: best = r
    a, S = unpack_cf1(best.x, m)
    return a, S, best.fun, time.time() - t0, np.array(results)

# ---------------- EM for general PH (Asmussen, Nerman & Olsson 1996) ----------------
def em_ph(x, alpha, S, tol=1e-11, maxit=20000, verbose=False, tol_abs=None):
    """Uncensored-data EM; matrix integrals in closed form via the eigenbasis of S
    (exact van Loan block exponential when eigenvalues nearly coincide)."""
    n, m = len(x), len(alpha); alpha = alpha.copy(); S = S.copy()
    ll_old = -np.inf; t0 = time.time(); trace = []
    for it in range(1, maxit + 1):
        s = -S @ np.ones(m)
        eta, V = np.linalg.eig(S); Vi = np.linalg.inv(V)
        gaps = np.abs(eta[:, None] - eta[None, :]) + np.eye(m)
        E = np.exp(np.outer(x, eta))                                  # n x m
        if gaps.min() > 1e-6 * np.abs(eta).max() and np.linalg.cond(V) < 1e8:
            a = (alpha @ V) * E @ Vi                                  # n x m   alpha e^{Sx}
            b = (E * (Vi @ s)) @ V.T                                  # n x m   e^{Sx} s
            G = Vi @ np.outer(s, alpha) @ V
            dE = (E[:, :, None] - E[:, None, :]) / (eta[:, None] - eta[None, :] + np.eye(m))
            diag = x[:, None] * E
            dE[:, np.arange(m), np.arange(m)] = diag
            H = dE * G[None]
            C = np.real(V[None] @ H @ Vi[None])                       # n x m x m
            a, b = np.real(a), np.real(b)
        else:
            big = np.zeros((2 * m, 2 * m)); big[:m, :m] = S; big[m:, m:] = S; big[:m, m:] = np.outer(s, alpha)
            Eb = expm(x[:, None, None] * big[None]); eS = Eb[:, :m, :m]
            a = np.einsum("j,njk->nk", alpha, eS); b = eS @ s; C = Eb[:, :m, m:]
        f = np.maximum(a @ s, 1e-300)
        ll = np.sum(np.log(f)); trace.append(ll)
        Bv = (alpha * b / f[:, None]).sum(0)
        Z = (np.diagonal(C, axis1=1, axis2=2) / f[:, None]).sum(0)
        N = (S[None] * np.transpose(C, (0, 2, 1)) / f[:, None, None]).sum(0)
        Nexit = (a * s / f[:, None]).sum(0)
        alpha = Bv / n
        Snew = N / Z[:, None]; np.fill_diagonal(Snew, 0.0)
        Snew = np.maximum(Snew, 0.0)
        snew = Nexit / Z
        np.fill_diagonal(Snew, -(Snew.sum(1) + snew)); S = Snew
        thr = tol_abs if tol_abs is not None else tol * abs(ll)
        if it > 1 and abs(ll - ll_old) < thr: break
        ll_old = ll
    return alpha, S, ll, it, time.time() - t0, np.array(trace)

def random_ph(m, rng, scale=1.0):
    S = rng.uniform(0, 1, (m, m)); np.fill_diagonal(S, 0); s = rng.uniform(0, 1, m)
    np.fill_diagonal(S, -(S.sum(1) + s)); S = S * scale / np.mean(-np.diag(S))
    return rng.dirichlet(np.ones(m)), S

# ---------------- goodness of fit ----------------
def gof(alpha, S, x):
    xs = np.sort(x); n = len(xs); F = np.clip(ph_cdf(alpha, S, xs), 1e-15, 1 - 1e-15)
    i = np.arange(1, n + 1)
    D = max(np.max(i / n - F), np.max(F - (i - 1) / n))
    AD = -n - np.mean((2 * i - 1) * (np.log(F) + np.log(1 - F[::-1])))
    # upper-tail Anderson-Darling AU^2 (Ahmad, Sinclair & Spurr 1988)
    ADup = n / 2 - 2 * np.sum(F) - np.sum((2 - (2 * i - 1) / n) * np.log(1 - F))
    return D, AD, ADup

# ---------------- sparse triangular dense-route (acyclic S) ----------------
import scipy.sparse as sp_
from scipy.sparse.linalg import spsolve_triangular
from math import comb as _comb
def pwm_tri(alpha, S, r):
    """Theorem 1 by sparse back-substitution; exact for upper-triangular (acyclic) S."""
    m = S.shape[0]; Ss = sp_.csr_matrix(S); s = -S @ np.ones(m); one = np.ones(m)
    K = Ss; T = []
    av = alpha.copy(); w1 = np.array([1.0])
    for j in range(r + 1):
        if j > 0:
            K = (sp_.kron(K, sp_.identity(m)) + sp_.kron(sp_.identity(m ** j), Ss)).tocsr()
            av = np.kron(av, alpha); w1 = np.kron(w1, one)
        w = np.kron(w1, s)
        y = spsolve_triangular(K, w, lower=False); z = spsolve_triangular(K, y, lower=False)
        T.append(av @ z)
    T = np.array(T)
    return np.array([sum((-1) ** j * _comb(q, j) * T[j] for j in range(q + 1)) for q in range(r + 1)])

_ph_lm_old = ph_lm
def ph_lm(alpha, S, R):
    if np.allclose(S, np.triu(S)):
        eta, V = np.linalg.eig(S)
        if np.finfo(float).eps * np.linalg.cond(V) ** R < 1e-8:
            return lmoments_from_pwm(pwm_spectral(alpha, S, R - 1))
        return lmoments_from_pwm(pwm_tri(alpha, S, R - 1))
    return _ph_lm_old(alpha, S, R)

def pack_cf1(rates, alpha):
    rates = np.asarray(rates, float); inc = np.diff(np.r_[0.0, rates])
    al = np.maximum(np.asarray(alpha, float), 1e-8); logit = np.log(al / al[-1])[:-1]
    return np.r_[np.log(np.maximum(inc, 1e-12)), logit]

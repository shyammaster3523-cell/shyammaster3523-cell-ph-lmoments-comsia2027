"""L-moments of phase-type distributions via Kronecker algebra.

Dense route:    beta_r = sum_j (-1)^j C(r,j) alpha^{(x)(j+1)} (S^{[j+1]})^{-2} (1^{(x)j} (x) s)
Spectral route: same sum, with S = V diag(eta) V^{-1}.
"""
import numpy as np
from math import comb
from scipy.linalg import lu_factor, lu_solve


def kron_power_vec(v, k):
    out = np.array([1.0], dtype=v.dtype)
    for _ in range(k):
        out = np.kron(out, v)
    return out


def kron_sum_power(S, k):
    """S^{[k]} = S (+) ... (+) S (k terms), dimension m^k."""
    m = S.shape[0]
    K = S.copy()
    for i in range(1, k):
        K = np.kron(K, np.eye(m)) + np.kron(np.eye(m ** i), S)
    return K


def pwm_dense(alpha, S, r):
    """beta_0..beta_r by the dense Kronecker route (two LU solves per lifted system)."""
    alpha = np.asarray(alpha, float); S = np.asarray(S, float)
    m = S.shape[0]
    s = -S @ np.ones(m)
    one = np.ones(m)
    # T_j = alpha^{(j+1)} M_j^{-2} (1^{j} (x) s),   M_j = S^{[j+1]}
    T = []
    for j in range(r + 1):
        M = kron_sum_power(S, j + 1)
        w = np.kron(kron_power_vec(one, j), s)
        lu = lu_factor(M)
        z = lu_solve(lu, lu_solve(lu, w))
        T.append(kron_power_vec(alpha, j + 1) @ z)
    T = np.array(T)
    return np.array([sum((-1) ** j * comb(q, j) * T[j] for j in range(q + 1)) for q in range(r + 1)])


def pwm_spectral(alpha, S, r, return_cond=False):
    """beta_0..beta_r by the spectral route (requires diagonalisable S)."""
    alpha = np.asarray(alpha, float); S = np.asarray(S, float)
    m = S.shape[0]
    eta, V = np.linalg.eig(S)
    Vi = np.linalg.inv(V)
    a = alpha @ V
    b = Vi @ np.ones(m)
    c = Vi @ (-S @ np.ones(m))
    ab, ac = a * b, a * c
    T = []
    W = np.array([1.0 + 0j]); E = np.array([0.0 + 0j])
    for j in range(r + 1):
        # tuples (i1..ij) weighted by a*b, last index weighted by a*c
        Wj = np.kron(W, ac); Ej = np.add.outer(E, eta).ravel()
        T.append(np.sum(Wj / Ej ** 2).real)
        W = np.kron(W, ab); E = np.add.outer(E, eta).ravel()
    T = np.array(T)
    beta = np.array([sum((-1) ** j * comb(q, j) * T[j] for j in range(q + 1)) for q in range(r + 1)])
    if return_cond:
        return beta, np.linalg.cond(V)
    return beta


def lmoments_from_pwm(beta):
    """lambda_1..lambda_{R} from beta_0..beta_{R-1} (shifted Legendre weights)."""
    R = len(beta)
    lam = []
    for k in range(R):
        lam.append(sum((-1) ** (k - j) * comb(k, j) * comb(k + j, j) * beta[j] for j in range(k + 1)))
    return np.array(lam)


def ph_lmoments(alpha, S, R, method="dense"):
    f = pwm_dense if method == "dense" else pwm_spectral
    return lmoments_from_pwm(f(alpha, S, R - 1))


def sample_lmoments(x, R):
    x = np.sort(np.asarray(x, float)); n = len(x)
    i = np.arange(1, n + 1)
    b = []
    for r in range(R):
        w = np.ones(n)
        for k in range(1, r + 1):
            w *= (i - k) / (n - k)
        b.append(np.mean(w * x))
    return lmoments_from_pwm(np.array(b))


def cf1(rates, alpha):
    """CF-1 (Cumani) generator: phase i -> i+1 at rate l_i, last phase absorbs."""
    m = len(rates)
    S = -np.diag(rates).astype(float)
    for i in range(m - 1):
        S[i, i + 1] = rates[i]
    return np.asarray(alpha, float), S

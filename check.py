import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)   # results (JSON) and figures are read/written in this folder
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from phlm import *
# exponential check
g=0.7; lam=ph_lmoments([1.0],np.array([[-g]]),4)
print("Exp:",lam[1]*2*g, lam[2]/lam[1], lam[3]/lam[1])
# quadrature check on random PH
rng=np.random.default_rng(1); worst=0
for t in range(20):
    m=rng.integers(2,5); S=rng.uniform(0,1,(m,m)); np.fill_diagonal(S,0)
    S-=np.diag(S.sum(1)+rng.uniform(0.2,1.5,m)); al=rng.dirichlet(np.ones(m)); s=-S@np.ones(m)
    bd=pwm_dense(al,S,5); bs=pwm_spectral(al,S,5)
    F=lambda x:1-al@expm(S*x)@np.ones(m); f=lambda x:al@expm(S*x)@s
    bq=[quad(lambda x:x*F(x)**r*f(x),0,np.inf,epsabs=0,epsrel=1e-13,limit=400)[0] for r in range(6)]
    worst=max(worst,np.max(np.abs(bd-bq)/np.abs(bq)),np.max(np.abs(bs-bd)/np.abs(bd)))
print("max rel err dense/quad/spectral:",worst)
# coxian-2 example from paper
print("Coxian-2 example L-moments:", ph_lmoments([1,0],np.array([[-2.,2],[0,-1]]),4))

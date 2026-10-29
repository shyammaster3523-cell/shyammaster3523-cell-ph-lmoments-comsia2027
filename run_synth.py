import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)   # results (JSON) and figures are read/written in this folder
import sys, json, numpy as np; from fitlib import *
from scipy.stats import kstwobign
def sim_ph(al,S,n,rng):
    m=len(al); s=-S@np.ones(m); out=np.empty(n)
    for i in range(n):
        st=rng.choice(m,p=al); t=0.0
        while True:
            q=-S[st,st]; t+=rng.exponential(1/q); pr=np.r_[np.maximum(S[st],0)*(np.arange(m)!=st),s[st]]/q
            nx=rng.choice(m+1,p=pr/pr.sum())
            if nx==m: break
            st=nx
        out[i]=t
    return out
rng=np.random.default_rng(2027); res={}
S=np.array([[-2.5,1.5,0],[0,-0.8,0.3],[0,0,-0.15]]); al=np.array([.7,.2,.1])
H=(np.array([.85,.15]),np.diag([-1.2,-0.05]))
for nm,(a0,S0),m in [("coxian3",(al,S),3),("hyperexp2",H,2)]:
    x=sim_ph(a0,S0,3000,rng); mu=x.mean(); z=x/mu; R=2*m-1; lam=sample_lmoments(z,R)
    a,Sf,fv,tt,allf=fit_cf1(lam,m,"L",starts=10,seed=1)
    D,AD,ADu=gof(a,Sf,z); p=kstwobign.sf(D*np.sqrt(len(z)))
    true=ph_lm(a0,S0/1.0,R)/mu; fit=ph_lm(a,Sf,R)
    res[nm]=dict(D=D,p=p,AD=AD,maxerr_true=float(np.max(np.abs(fit/true-1))*100),obj=fv)
    print(nm,f"KS D={D:.4f} p={p:.3f} AD={AD:.2f} max rel. diff fitted vs TRUE L-moments={res[nm]['maxerr_true']:.2f}% obj={fv:.1e}")
json.dump(res,open("synth.json","w"),indent=1)

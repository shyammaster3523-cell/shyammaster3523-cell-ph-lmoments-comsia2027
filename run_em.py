import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)   # results (JSON) and figures are read/written in this folder
import sys, json, os, numpy as np, time; from fitlib import *
data=sys.argv[1]; m=int(sys.argv[2]); which=sys.argv[3:]
x=load(data); mu=x.mean(); z=x/mu; n=len(z); lam=sample_lmoments(z,5)
fits=json.load(open(f"{data}_fits.json")); shift=n*np.log(mu)
TOLABS=1e-11*abs(fits[f"L{m}"]["ll"])   # relative 1e-11 on the original-scale log-likelihood
def start(name):
    if name.startswith("rand"):
        rng=np.random.default_rng(int(name[4:])); return random_ph(m,rng)
    if name=="quant":   # quantile-split hyperexponential (common heuristic EM start)
        g=np.array_split(np.sort(z),m); return np.full(m,1/m),-np.diag([1/gg.mean() for gg in g])
    if name in ("Lmom","Cmom"):
        f=fits[("L" if name=="Lmom" else "C")+str(m)]; return np.array(f["alpha"]),np.array(f["S"])
f=f"{data}_em{m}.json"; out=json.load(open(f)) if os.path.exists(f) else {}
for w in which:
    a0,S0=start(w); a,S,ll,it,tt,tr=em_ph(z,a0,S0,maxit=20000,tol_abs=TOLABS)
    D,AD,ADu=gof(a,S,z); err=(ph_lm(a,S,5)/lam-1)*100
    out[w]=dict(ll=ll-shift,it=it,time=tt,D=D,AD=AD,ADu=ADu,lerr=err.tolist(),alpha=a.tolist(),S=(np.array(S)/mu).tolist(),ll0=float(tr[0]-shift))
    json.dump(out,open(f,"w"),indent=1)
    print(f"{w}: ll={ll-shift:.2f} it={it} t={tt:.0f}s KS={D:.4f} Lerr={np.round(err,2)}",flush=True)

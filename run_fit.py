import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)   # results (JSON) and figures are read/written in this folder
import sys, json, os, numpy as np, time; from fitlib import *
data,kind,m,starts=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
x=load(data); mu=x.mean(); z=x/mu; R=2*m-1
lam=sample_lmoments(z,max(R,5))
tgt=lam[:R] if kind=="L" else np.array([np.mean(z**k) for k in range(1,R+1)])
a,S,fv,tt,allf=fit_cf1(tgt,m,kind,starts=starts,seed=m)
lm=ph_lm(a,S,5); err=(lm/lam[:5]-1)*100; D,AD,ADu=gof(a,S,z); ll=ph_logpdf(a,S,z).sum()-len(z)*np.log(mu)
res=dict(alpha=a.tolist(),rates=(-np.diag(S)/mu).tolist(),obj=fv,time=tt,starts=starts,lerr=err.tolist(),D=D,AD=AD,ADu=ADu,ll=ll,
         share_best=float(np.mean(allf<=fv*(1+1e-3)+1e-12)),S=S.tolist())
f=f"{data}_fits.json"; old=json.load(open(f)) if os.path.exists(f) else {}; old[f"{kind}{m}"]=res; json.dump(old,open(f,"w"),indent=1)
print(f"{data} {kind}{m}: obj={fv:.2e} t={tt:.0f}s Lerr={np.round(err,2)} KS={D:.4f} AD={AD:.1f} ADup={ADu:.2f} ll={ll:.2f} best-share={res['share_best']:.2f}")

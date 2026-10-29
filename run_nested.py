import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)   # results (JSON) and figures are read/written in this folder
import sys, json, numpy as np; from fitlib import *
data,kind,m,extra=sys.argv[1],sys.argv[2],int(sys.argv[3]),int(sys.argv[4])
x=load(data); mu=x.mean(); z=x/mu; R=2*m-1; lam=sample_lmoments(z,max(R,5))
tgt=lam[:R] if kind=="L" else np.array([np.mean(z**k) for k in range(1,R+1)])
prev=json.load(open(f"{data}_fits.json"))[f"{kind}{m-1}"]
r0=np.array(prev["rates"])*mu; a0=np.array(prev["alpha"])
inits=[]
for fac in (2.0,4.0,0.5):
    for pos in ("end","start"):
        if pos=="end": rates=np.r_[r0,r0[-1]*fac if fac>1 else r0[-1]*1.5]; al=np.r_[a0,1e-4]
        else: rates=np.r_[r0[0]*min(fac,0.9),r0]; al=np.r_[1e-4,a0]
        al=al/al.sum(); inits.append(pack_cf1(np.sort(rates),al))
a,S,fv,tt,allf=fit_cf1(tgt,m,kind,starts=extra,seed=10+m,init=inits[:int(sys.argv[5]) if len(sys.argv)>5 else 6])
lm=ph_lm(a,S,5); err=(lm/lam[:5]-1)*100; D,AD,ADu=gof(a,S,z); ll=ph_logpdf(a,S,z).sum()-len(z)*np.log(mu)
f=f"{data}_fits.json"; old=json.load(open(f))
cur=old.get(f"{kind}{m}")
res=dict(alpha=a.tolist(),rates=(-np.diag(S)/mu).tolist(),obj=fv,time=tt,starts=len(allf),lerr=err.tolist(),D=D,AD=AD,ADu=ADu,ll=ll,S=S.tolist(),
         share_best=float(np.mean(allf<=fv*(1+1e-3)+1e-12)),init="nested")
if cur: res["random_only"]=dict(obj=cur["obj"],time=cur["time"],starts=cur["starts"])
old[f"{kind}{m}"]=res; json.dump(old,open(f,"w"),indent=1)
print(f"{data} {kind}{m} nested: obj={fv:.2e} t={tt:.0f}s Lerr={np.round(err,3)} KS={D:.4f} AD={AD:.2f} ADup={ADu:.2f} ll={ll:.2f} alpha={np.round(a,4)} rates={np.round(-np.diag(S)/mu*1e4,3)}")

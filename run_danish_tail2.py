import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)   # results (JSON) and figures are read/written in this folder
import sys, json, numpy as np; from fitlib import *
from scipy import stats
from scipy.optimize import brentq
x=load("danish"); y=load("danishx"); n=len(x); xs=np.sort(x)
lam=sample_lmoments(y,4); print("excess tau,t3,t4:",np.round([lam[1]/lam[0],lam[2]/lam[1],lam[3]/lam[1]],4),
      " bound(5t-2)/(3t)=%.4f"%((5*lam[1]/lam[0]-2)/(3*lam[1]/lam[0])))
lx=sample_lmoments(x,4); print("raw tau,t3,t4:",np.round([lx[1]/lx[0],lx[2]/lx[1],lx[3]/lx[1]],4))
H={k:float(1/np.mean(np.log(xs[-k:]/xs[-k-1]))) for k in (50,100,200)}
emp={p:float(np.quantile(x,p)) for p in (0.99,0.995,0.999)}
out={"hill":H,"emp":emp,"rows":{}}
def row(name,cdf_x,logpdf_x,k,ppf_x):
    F=np.clip(cdf_x(xs),1e-15,1-1e-15); i=np.arange(1,n+1)
    D=max(np.max(i/n-F),np.max(F-(i-1)/n)); AD=-n-np.mean((2*i-1)*(np.log(F)+np.log(1-F[::-1])))
    ADu=n/2-2*np.sum(F)-np.sum((2-(2*i-1)/n)*np.log(1-F)); ll=float(np.sum(logpdf_x(x)))
    q={p:float(ppf_x(p)) for p in (0.99,0.995,0.999)}
    out["rows"][name]=dict(k=k,ll=ll,aic=2*k-2*ll,D=D,AD=AD,ADu=ADu,var=q)
    print(f"{name:22s} k={k} ll={ll:9.2f} AIC={2*k-2*ll:8.2f} KS={D:.4f} AD={AD:7.2f} ADup={ADu:6.2f} VaR99={q[0.99]:6.2f} 99.5={q[0.995]:6.2f} 99.9={q[0.999]:7.2f}")
print("empirical VaR:",{p:round(v,2) for p,v in emp.items()},"Hill:",{k:round(v,2) for k,v in H.items()})
for nm,dist,kk,shift in [("lognormal",stats.lognorm,2,0),("Weibull",stats.weibull_min,2,0)]:
    p=dist.fit(x,floc=0); d=dist(*p); row(nm,d.cdf,d.logpdf,kk,d.ppf)
a=n/np.sum(np.log(x)); d=stats.pareto(a,0,1.0); row("Pareto (x_m=1)",d.cdf,d.logpdf,1,d.ppf); out["pareto_alpha"]=a
c,_,sc=stats.genpareto.fit(y,floc=0); g=stats.genpareto(c,0,sc)
row("GPD on x-1",lambda t:g.cdf(t-1),lambda t:g.logpdf(t-1),2,lambda p:1+g.ppf(p)); out["gpd"]=dict(xi=c,beta=sc)
mu=y.mean(); fx=json.load(open("danishx_fits.json")); ex=json.load(open("danishx_em3.json")); f0=json.load(open("danish_fits.json")); e0=json.load(open("danish_em3.json"))
def phrow(name,al,S,k,shift):
    al=np.array(al); S=np.array(S)
    cdf=lambda t: ph_cdf(al,S,np.maximum(t-shift,0)); lp=lambda t: ph_logpdf(al,S,t-shift)
    ppf=lambda p: shift+brentq(lambda t: ph_cdf(al,S,t)[0]-p,1e-9,1e5)
    row(name,cdf,lp,k,ppf)
phrow("PH3 L-mom, raw",f0["L3"]["alpha"],np.array(f0["L3"]["S"])/x.mean(),5,0)
phrow("PH3 EM, raw",e0["rand0"]["alpha"],e0["rand0"]["S"],8,0)
phrow("PH3 L-mom, x-1",fx["L3"]["alpha"],np.array(fx["L3"]["S"])/mu,5,1)
phrow("PH3 EM, x-1",ex["rand0"]["alpha"],ex["rand0"]["S"],8,1)
json.dump(out,open("danish_tail.json","w"),indent=1,default=float)

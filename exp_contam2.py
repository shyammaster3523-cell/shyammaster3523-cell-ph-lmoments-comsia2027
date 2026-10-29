import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)   # results (JSON) and figures are read/written in this folder
import numpy as np, json, time, sys
from scipy.optimize import minimize
from phlm import pwm_spectral as pwm_dense, lmoments_from_pwm, sample_lmoments
P,G1,G2=0.80,0.005,0.001; n=500; LO,HI=1e-5,1.0
B=int(sys.argv[1]) if len(sys.argv)>1 else 200
def hyp(p,g1,g2): return np.array([p,1-p]),np.diag([-g1,-g2])
def lm3(p,g1,g2): al,S=hyp(p,g1,g2); return lmoments_from_pwm(pwm_dense(al,S,2))
def cm3(p,g1,g2): return np.array([f*(p/g1**k+(1-p)/g2**k) for k,f in ((1,1),(2,2),(3,6))])
def unpack(t): return 1/(1+np.exp(-t[0])),np.exp(t[1]),np.exp(t[2])
bnds=[(-9,9),(np.log(LO),np.log(HI)),(np.log(LO),np.log(HI))]
starts=[(0.0,-4,-7),(1.4,-5,-7.5),(-1.4,-4.5,-6.5),(2.0,-3,-8),(0.5,-6,-9)]
def fit(target,fun):
    best=None
    for s in starts:
        obj=lambda t:np.sum((fun(*unpack(t))/target-1)**2)
        r=minimize(obj,np.array(s,float),method="L-BFGS-B",bounds=bnds,options=dict(ftol=1e-14,gtol=1e-10,maxiter=2000))
        if best is None or r.fun<best.fun: best=r
    return unpack(best.x),best
def em(x,tol=1e-8,maxit=5000):
    m=x.mean(); p,g1,g2=0.5,2/m,0.5/m
    ll0=-np.inf
    for it in range(maxit):
        d1=p*g1*np.exp(-g1*x); d2=(1-p)*g2*np.exp(-g2*x); tot=d1+d2; w=d1/tot
        ll=np.sum(np.log(tot)); 
        p=w.mean(); g1=w.sum()/np.sum(w*x); g2=(1-w).sum()/np.sum((1-w)*x)
        if abs(ll-ll0)<tol*abs(ll): break
        ll0=ll
    return (p,g1,g2),it+1
def err(est):
    g=sorted([est[1],est[2]],reverse=True)
    se=(g[0]-G1)**2+(g[1]-G2)**2
    gross=abs(g[0]/G1-1)>1 or abs(g[1]/G2-1)>1
    pinned=any(min(abs(np.log(v/LO)),abs(np.log(v/HI)))<0.01 for v in g)
    return se,gross,pinned
EPS=[0.0,0.01,0.02,0.05,0.10,0.20]; MAG=[0.25,1.0,4.0]
cells=[(e,k) for k in MAG for e in EPS if not (e==0 and k!=1.0)]
lo,hi=int(sys.argv[2]),int(sys.argv[3])
import os
for eps,k in cells[lo:hi]:
    rng=np.random.default_rng(int(1000*eps)+int(10*k))
    rec={m:[] for m in ("L","C","EM")}; its=[]
    for b in range(B):
        z=rng.random(n)<P; x=np.where(z,rng.exponential(1/G1,n),rng.exponential(1/G2,n))
        nc=int(round(eps*n))
        if nc: x[rng.choice(n,nc,replace=False)]=rng.uniform(5000*k,20000*k,nc)
        eL,_=fit(sample_lmoments(x,3),lm3)
        eC,_=fit(np.array([np.mean(x),np.mean(x**2),np.mean(x**3)]),cm3)
        eE,it=em(x); its.append(it)
        for m,e in (("L",eL),("C",eC),("EM",eE)): rec[m].append(err(e))
    row=dict(eps=eps,mag=k)
    for m in rec:
        a=np.array(rec[m],float)
        row[m]=dict(med=float(np.median(a[:,0])),q1=float(np.quantile(a[:,0],.25)),q3=float(np.quantile(a[:,0],.75)),gross=float(a[:,1].mean()),pinned=float(a[:,2].mean()))
    with open("contam.jsonl","a") as f: f.write(json.dumps(row)+"\n")
    print(eps,k,{m:"%.2e/%.0f%%"%(row[m]["med"],100*row[m]["gross"]) for m in rec},flush=True)

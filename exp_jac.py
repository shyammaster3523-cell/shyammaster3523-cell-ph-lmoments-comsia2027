import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)   # results (JSON) and figures are read/written in this folder
import numpy as np, json
from phlm import *
# theta = (log l1, log l2, log l3, a1, a2), a3 = 1-a1-a2 ; output = log lambda_1..lambda_5 (lambda_r>0 here)
def G(th):
    l=np.exp(th[:3]); a=[th[3],th[4],1-th[3]-th[4]]
    al,S=cf1(l,a); return ph_lmoments(al,S,5,"dense")
def J(th,h=1e-6):
    g0=G(th); Jm=np.zeros((5,5))
    for k in range(5):
        e=np.zeros(5); e[k]=h
        Jm[:,k]=(G(th+e)-G(th-e))/(2*h)
    # relative sensitivity: d log lambda / d theta
    return Jm/g0[:,None]
res={}
# 1) AutoClaims-type estimate (from the paper), scaled
l=np.array([2.63004,11.7626,50.0929])*1e-4; base=np.r_[np.log(l),0.210376,0.789624]
Jm=J(base); sv=np.linalg.svd(Jm,compute_uv=False); res["autoclaims"]=dict(cond=float(sv[0]/sv[-1]),smin=float(sv[-1]))
print("AutoClaims estimate: cond(J)=%.3g  sigma_min=%.3g"%(sv[0]/sv[-1],sv[-1]))
# perturbation: 1% relative noise on lambda -> induced change in theta (first-order)
Jinv=np.linalg.pinv(Jm); dl=0.01*np.ones(5)/np.sqrt(5)
print("theta change per 1% (norm) L-moment perturbation:", np.round(np.abs(Jinv)@np.full(5,0.01),3))
# 2) coalescing rates l2 -> l1
res["coalesce"]=[]
for gap in [1.0,0.3,0.1,0.03,0.01,0.003]:
    th=np.r_[np.log([1.0,1.0+gap,4.0]),0.4,0.4]; sv=np.linalg.svd(J(th),compute_uv=False)
    res["coalesce"].append((gap,float(sv[0]/sv[-1]))); print("rate gap %.3f cond(J)=%.3g"%(gap,sv[0]/sv[-1]))
# 3) vanishing initial weight a1 -> 0 (with a2 fixed)
res["weight"]=[]
for a1 in [0.3,0.1,0.03,0.01,0.001]:
    th=np.r_[np.log([1.0,2.0,4.0]),a1,0.4]; sv=np.linalg.svd(J(th),compute_uv=False)
    res["weight"].append((a1,float(sv[0]/sv[-1]))); print("a1=%.3f cond(J)=%.3g"%(a1,sv[0]/sv[-1]))
# 4) random grid
rng=np.random.default_rng(0); cs=[]
for _ in range(2000):
    l=np.sort(rng.uniform(np.log(0.1),np.log(10),3)); a=rng.dirichlet(np.ones(3))
    sv=np.linalg.svd(J(np.r_[l,a[0],a[1]]),compute_uv=False); cs.append(sv[0]/sv[-1])
cs=np.array(cs); q=np.quantile(cs,[.5,.9,.99]); res["grid"]=dict(q50=q[0],q90=q[1],q99=q[2],max=cs.max())
print("random grid cond(J): median %.3g, 90%% %.3g, 99%% %.3g, max %.3g"%(*q,cs.max()))
json.dump(res,open("jac.json","w"),indent=1,default=float)

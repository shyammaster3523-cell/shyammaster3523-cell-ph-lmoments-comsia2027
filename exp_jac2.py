import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)   # results (JSON) and figures are read/written in this folder
import numpy as np, json
from jacfun import J
def cond(l,a): sv=np.linalg.svd(J(np.r_[np.log(l),a[0],a[1]]),compute_uv=False); return sv[0]/sv[-1]
def numer_at_poles(l,a):
    # CF-1 Laplace numerator relative magnitude at each pole s=-l_k (0 => cancellation, order drops)
    l=np.asarray(l); out=[]
    for lk in l:
        s=-lk; tot=0
        for i in range(3):
            p=np.prod([l[k]/(l[k]+s) if l[k]!=lk else l[k] for k in range(i,3)])
            tot+=a[i]*p if any(l[k]==lk for k in range(i,3)) else 0
        out.append(tot)
    return out
R={}
print("(a) coalescing rates, alpha=(0.5,0.3,0.2), l=(1,1+g,4)")
R["coalesce"]=[(g,cond([1,1+g,4],[.5,.3,.2])) for g in [1,0.3,0.1,0.03,0.01,0.003,0.001]]
for g,c in R["coalesce"]: print("  g=%g cond=%.3g"%(g,c))
print("(b) first-phase weight a1->0, l=(1,3,5), alpha=(a1,0.5,0.5-a1)")
R["weight"]=[(a,cond([1,3,5],[a,.5,.5-a])) for a in [0.3,0.1,0.03,0.01,0.003,0.001]]
for a,c in R["weight"]: print("  a1=%g cond=%.3g"%(a,c))
print("(c) approach to pole-zero cancellation, l=(1,2,4), alpha=(0.4,0.4+d,0.2-d)")
R["cancel"]=[(d,cond([1,2,4],[.4,.4+d,.2-d])) for d in [0.1,0.03,0.01,0.003,0.001,1e-4]]
for d,c in R["cancel"]: print("  d=%g cond=%.3g"%(d,c))
json.dump(R,open("jac2.json","w"),indent=1)

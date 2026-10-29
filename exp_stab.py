import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)   # results (JSON) and figures are read/written in this folder
import numpy as np, json
import mpmath as mp
from phlm import *
mp.mp.dps=40
def pwm_mp(alpha,S,r):
    # high-precision dense route as ground truth
    m=len(alpha); A=mp.matrix(S); al=[mp.mpf(a) for a in alpha]
    s=[-sum(A[i,j] for j in range(m)) for i in range(m)]
    from math import comb
    T=[]
    for j in range(r+1):
        M=np.array(kron_sum_power(np.array(S,float),j+1))  # entries are exact sums of inputs
        Mm=mp.matrix(M.tolist())
        w=[1]; 
        for _ in range(j): w=[x*1 for x in w for _ in range(m)]
        w=[x*si for x in w for si in s]
        av=[1]
        for _ in range(j+1): av=[x*a for x in av for a in al]
        z=mp.lu_solve(Mm,mp.lu_solve(Mm,mp.matrix(w)))
        T.append(sum(av[i]*z[i] for i in range(len(av))))
    return [sum((-1)**j*comb(q,j)*T[j] for j in range(q+1)) for q in range(r+1)]
al=[0.5,0.3,0.2]; out=[]
for gap in [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]:
    a,S=cf1([1.0,1.0+gap,3.0],al)
    bs,cv=pwm_spectral(a,S,4,return_cond=True); bd=pwm_dense(a,S,4); bm=pwm_mp(a,S,4)
    es=[abs(bs[r]-float(bm[r]))/abs(float(bm[r])) for r in range(5)]
    ed=[abs(bd[r]-float(bm[r]))/abs(float(bm[r])) for r in range(5)]
    out.append(dict(gap=gap,condV=cv,spec=es,dense=ed))
    print("gap %.0e cond(V) %.2e | spectral r=0..4: %s | dense max %.1e"%(gap,cv," ".join("%.1e"%e for e in es),max(ed)))
json.dump(out,open("stab.json","w"),indent=1)
# exact repeated eigenvalue (Erlang-3)
a,S=cf1([1.0,1.0,1.0],[1,0,0])
try:
    b=pwm_spectral(a,S,2); print("Erlang spectral:",b, "dense:",pwm_dense(a,S,2))
except Exception as e: print("Erlang spectral fails:",e)

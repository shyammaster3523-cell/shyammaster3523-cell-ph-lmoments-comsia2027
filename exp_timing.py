import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)   # results (JSON) and figures are read/written in this folder
import numpy as np, time, json
from phlm import *
rng=np.random.default_rng(7)
def randph(m):
    S=rng.uniform(0,1,(m,m)); np.fill_diagonal(S,0)
    S-=np.diag(S.sum(1)+rng.uniform(0.2,1.5,m)); return rng.dirichlet(np.ones(m)),S
def tm(f,rep):
    ts=[]
    for _ in range(rep):
        t=time.perf_counter(); f(); ts.append(time.perf_counter()-t)
    return float(np.median(ts))
res={"dense":{},"spectral":{}}
for m in [2,3,4,5,6,8]:
    al,S=randph(m)
    for r in range(0,7):
        N=m**(r+1)
        if N<=2500:
            res["dense"][f"{m},{r}"]=tm(lambda:pwm_dense(al,S,r),3 if N>500 else 9)
        if N<=3e6:
            res["spectral"][f"{m},{r}"]=tm(lambda:pwm_spectral(al,S,r),3 if N>1e5 else 9)
json.dump(res,open("timing.json","w"),indent=1)
# slopes in log m for fixed r, using largest available m
for meth in res:
    for r in range(7):
        pts=[(int(k.split(',')[0]),v) for k,v in res[meth].items() if int(k.split(',')[1])==r]
        pts=[p for p in pts if (p[0]**(r+1))>=200]
        if len(pts)>=2:
            x=np.log([p[0] for p in pts]); y=np.log([p[1] for p in pts])
            print(meth,"r=",r,"slope=%.2f"%np.polyfit(x,y,1)[0],"theory=",3*(r+1) if meth=="dense" else r+1, "n=",len(pts))

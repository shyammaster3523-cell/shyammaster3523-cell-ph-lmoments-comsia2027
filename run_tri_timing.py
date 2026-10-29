import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)   # results (JSON) and figures are read/written in this folder
import sys, time, json, numpy as np; from fitlib import pwm_tri, cf1
res={}
for m in (2,3,4,6,8,10):
    a,S=cf1(np.linspace(1,3,m),np.ones(m)/m)
    for r in range(0,9):
        N=m**(r+1)
        if N>3e6: break
        t=time.perf_counter(); pwm_tri(a,S,r); dt=time.perf_counter()-t
        res[f"{m},{r}"]=dt
json.dump(res,open("tri_timing.json","w"),indent=1)
pts=sorted((int(k.split(',')[0])**(int(k.split(',')[1])+1),v) for k,v in res.items())
big=[p for p in pts if p[0]>=2e4]
print("points",len(pts),"max N",max(p[0] for p in pts),"t=%.2fs"%max(pts)[1], "slope(N>=2e4)=%.2f"%np.polyfit(np.log([p[0] for p in big]),np.log([p[1] for p in big]),1)[0])

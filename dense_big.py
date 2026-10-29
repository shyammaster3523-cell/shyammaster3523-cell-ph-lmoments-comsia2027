import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)   # results (JSON) and figures are read/written in this folder
import numpy as np,time,json
from phlm import *
rng=np.random.default_rng(3); res=json.load(open("timing.json"))
for m,r in [(4,5),(16,2),(8,3)]:
    S=rng.uniform(0,1,(m,m)); np.fill_diagonal(S,0); S-=np.diag(S.sum(1)+rng.uniform(.2,1.5,m)); al=rng.dirichlet(np.ones(m))
    t=time.perf_counter(); pwm_dense(al,S,r); res["dense"][f"{m},{r}"]=time.perf_counter()-t
    print(m,r,m**(r+1),res["dense"][f"{m},{r}"])
json.dump(res,open("timing.json","w"),indent=1)

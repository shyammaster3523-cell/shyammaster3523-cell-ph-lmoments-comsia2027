import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)   # results (JSON) and figures are read/written in this folder
import json, numpy as np
rows=[]
for data,lab in [("auto","AutoClaims"),("danish","Danish (raw)"),("danishx","Danish (x-1)")]:
    F=json.load(open(f"{data}_fits.json"))
    for m in (2,3,4):
        for kind,name in (("L","L-moments"),("C","Classical")):
            k=f"{kind}{m}"
            if k in F:
                f=F[k]; rows.append((lab,m,name,max(abs(np.array(f["lerr"]))),f["D"],f["AD"],f["ll"],f["time"],f"{f['share_best']:.2f}"))
        try: E=json.load(open(f"{data}_em{m}.json"))
        except: continue
        lls=np.array([v["ll"] for v in E.values()]); best=lls.max(); kk=[k for k,v in E.items() if v["ll"]==best][0]; v=E[kk]
        rand=[e for k,e in E.items() if k.startswith("rand")]
        rel=np.mean([abs(e["ll"]-best)<0.05 for e in rand]); its=[e["it"] for e in rand]; tm=np.median([e["time"] for e in rand])
        rows.append((lab,m,"EM",max(abs(np.array(v["lerr"]))),v["D"],v["AD"],v["ll"],tm,f"{rel:.2f}",int(np.median(its)),min(its),max(its)))
for r in rows: print(r)
json.dump(rows,open("consolidated.json","w"),indent=1,default=float)
A=json.load(open("auto_em3.json"))
print({k:(v["it"],round(v["time"],1),round(v["ll"],2)) for k,v in A.items()})

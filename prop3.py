import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)   # results (JSON) and figures are read/written in this folder
import sympy as sp, numpy as np, mpmath as mp, json
from math import comb
a,u=sp.symbols('a u',positive=True)
# order-2 acyclic, CF-1 canonical, scaled so l2=1: S=[[-u,u],[0,-1]], alpha=(a,1-a), 0<u<=1
S=sp.Matrix([[-u,u],[0,-1]]); al=sp.Matrix([[a,1-a]]); one=sp.Matrix([1,1]); s=-S*one
def ksum(S,k):
    K=S
    for i in range(1,k): K=sp.kronecker_product(K,sp.eye(2))+sp.kronecker_product(sp.eye(2**i),S)
    return K
def kp(v,k):
    out=sp.Matrix([[1]]) if v.shape[0]==1 else sp.Matrix([1])
    for _ in range(k): out=sp.kronecker_product(out,v)
    return out
T=[]
for j in range(3):
    M=ksum(S,j+1); w=sp.kronecker_product(kp(one,j),s)
    z=M.LUsolve(M.LUsolve(w)); T.append(sp.simplify((kp(al,j+1)*z)[0]))
B=[sp.simplify(sum((-1)**j*comb(q,j)*T[j] for j in range(q+1))) for q in range(3)]
l1=B[0]; l2=2*B[1]-B[0]; l3=6*B[2]-6*B[1]+B[0]
tau=sp.factor(sp.simplify(l2/l1)); tau3=sp.factor(sp.simplify(l3/l2))
print("lambda1 =",sp.factor(l1)); print("tau  =",tau); print("tau3 =",tau3)
open("prop3_sym.txt","w").write(f"lambda1={sp.latex(sp.factor(l1))}\ntau={sp.latex(tau)}\ntau3={sp.latex(tau3)}\n")
ft=sp.lambdify((a,u),tau,"numpy"); f3=sp.lambdify((a,u),tau3,"numpy")
# dense double-precision grid incl. near the edges
A=np.concatenate([np.linspace(0,1,1201)[1:-1],1-np.logspace(-8,-1,200),np.logspace(-8,-1,200)])
U=np.concatenate([np.linspace(0,1,1201)[1:],np.logspace(-8,-1,200)])
AA,UU=np.meshgrid(A,U); t=ft(AA,UU); t3=f3(AA,UU)
mask=t>0.5; ex=t3[mask]-(5*t[mask]-2)/(3*t[mask])
print("grid points with tau>1/2:",mask.sum()," max(tau3-bound) double: %.3e"%ex.max())
# high-precision recheck of the 500 points closest to the bound
idx=np.argsort(-ex)[:500]; aa=AA[mask][idx]; uu=UU[mask][idx]
mp.mp.dps=50
ftm=sp.lambdify((a,u),tau,"mpmath"); f3m=sp.lambdify((a,u),tau3,"mpmath")
worst=max(f3m(mp.mpf(x),mp.mpf(y))-(5*ftm(mp.mpf(x),mp.mpf(y))-2)/(3*ftm(mp.mpf(x),mp.mpf(y))) for x,y in zip(aa,uu))
print("max excess at 50 digits over top-500:", mp.nstr(worst,6))
print("argmax point (a,u):",aa[0],uu[0])
# edge behaviour: is the sup approached as u->0 with a->? 
for uu0 in [1e-2,1e-4,1e-6]:
    tt=ft(A,uu0); tt3=f3(A,uu0); m=tt>0.5
    print("u=%g: max tau3-bound over a = %.3e at a=%.4g"%(uu0,(tt3[m]-(5*tt[m]-2)/(3*tt[m])).max(),A[m][np.argmax(tt3[m]-(5*tt[m]-2)/(3*tt[m]))]))
